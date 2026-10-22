import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

OPS        = "+-*/"
MAX_N      = 100
ALLOW_NEG  = True
TRAIN_POOL = 30000
SCRATCHPAD = True

TEST_PCT   = 15
TEST_PER_OP = 100

D_MODEL, N_LAYERS, N_HEADS = 192, 6, 6
HEAD_DIM = D_MODEL // N_HEADS
QK_NORM = True
STEPS, BATCH_SIZE, LR = 1500, 256, 3e-4
WARMUP, WD, EVAL_EVERY = 300, 0.1, 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); random.seed(0)

SCRATCHPAD = SCRATCHPAD and "*" in OPS

chars = list("0123456789+-*/=;")
PAD, EOS = len(chars), len(chars) + 1
stoi = {c: i for i, c in enumerate(chars)}
VOCAB = len(chars) + 2
BLOCK = 128 if SCRATCHPAD else 2 * len(str(MAX_N * MAX_N)) + 12

def compute(op, a, b):
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/": return a // b
    raise ValueError(op)

def rand_operand():
    return random.randint(-MAX_N, MAX_N) if ALLOW_NEG else random.randint(0, MAX_N)

def gen_raw(op):
    if op == "/":
        b = 0
        while b == 0:
            b = rand_operand()
        q = rand_operand()
        return b * q, b
    return rand_operand(), rand_operand()

def is_test(op, a, b):
    h = (a * 73856093) ^ (b * 19349663) ^ (ord(op) * 83492791)
    return (h % 100) < TEST_PCT

def gen_problem(want_test):
    while True:
        op = random.choice(OPS)
        a, b = gen_raw(op)
        if is_test(op, a, b) == want_test:
            return (op, a, b)

def build_mul_steps(a, b):
    A, B = abs(a), abs(b)
    sign = -1 if (a < 0) ^ (b < 0) else 1
    steps, terms, place = [], [], 1
    for ch in reversed(str(B)):
        d = int(ch)
        if d != 0:
            steps.append(f"{A}*{d * place}={A * d * place}")
            terms.append(A * d * place)
        place *= 10
    if not terms:
        steps.append(f"{A}*0=0"); terms = [0]
    running = terms[0]
    for t in terms[1:]:
        steps.append(f"{running}+{t}={running + t}")
        running += t
    return steps + [f"={sign * running}"], sign * running

def encode_example(op, a, b):
    r = compute(op, a, b)
    prompt = f"{a}{op}{b}="
    if SCRATCHPAD and op == "*":
        segments, _ = build_mul_steps(a, b)
        tail = ";".join(segments)
    else:
        tail = ("-" if r < 0 else "") + str(abs(r))[::-1]
    ids = [stoi[c] for c in prompt] + [stoi[c] for c in tail] + [EOS]
    assert len(ids) <= BLOCK, f"пример длиннее BLOCK={BLOCK}: {prompt}{tail}"
    plen = len(prompt)
    x, y = ids[:-1], ids[1:]
    for j in range(plen - 1):
        y[j] = -100
    return x, y

def make_batch(problems, bs):
    picks = [random.choice(problems) for _ in range(bs)]
    enc = [encode_example(*p) for p in picks]
    maxlen = max(len(x) for x, _ in enc)
    X = torch.full((bs, maxlen), PAD, dtype=torch.long)
    Y = torch.full((bs, maxlen), -100, dtype=torch.long)
    for i, (x, y) in enumerate(enc):
        X[i, :len(x)] = torch.tensor(x)
        Y[i, :len(y)] = torch.tensor(y)
    return X.to(DEVICE), Y.to(DEVICE)

seen = set()
train_pool = []
stall = 0
while len(train_pool) < TRAIN_POOL and stall < 30000:
    p = gen_problem(False)
    if p not in seen:
        seen.add(p); train_pool.append(p); stall = 0
    else:
        stall += 1
if len(train_pool) < TRAIN_POOL:
    print(f"[!] разных train-примеров всего {len(train_pool)} "
          f"(TRAIN_POOL={TRAIN_POOL} велик для OPS={OPS}/MAX_N={MAX_N}/ALLOW_NEG={ALLOW_NEG}). "
          f"Беру сколько есть. Для большего разнообразия подними MAX_N.")

test_by_op = {op: [] for op in OPS}
seen_t = set()
for op in OPS:
    stall = 0
    while len(test_by_op[op]) < TEST_PER_OP and stall < 30000:
        a, b = gen_raw(op)
        if is_test(op, a, b) and (op, a, b) not in seen_t:
            seen_t.add((op, a, b)); test_by_op[op].append((op, a, b)); stall = 0
        else:
            stall += 1
test_all = [p for op in OPS for p in test_by_op[op]]
print(f"OPS={OPS} MAX_N={MAX_N} ALLOW_NEG={ALLOW_NEG} | train {len(train_pool)} | "
      f"test {{{', '.join(f'{op}:{len(test_by_op[op])}' for op in OPS)}}}")

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)
        o = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return o.type_as(x) * self.weight

class Rotary(nn.Module):
    def __init__(self, hd, max_pos, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
        f = torch.outer(torch.arange(max_pos).float(), inv)
        emb = torch.cat((f, f), -1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
    def forward(self, T): return self.cos[:T], self.sin[:T]

def rotate_half(x):
    a, b = x.chunk(2, -1); return torch.cat((-b, a), -1)
def apply_rope(q, k, cos, sin):
    cos, sin = cos[None, None], sin[None, None]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.rot = Rotary(HEAD_DIM, BLOCK)
        self.qn = RMSNorm(HEAD_DIM) if QK_NORM else nn.Identity()
        self.kn = RMSNorm(HEAD_DIM) if QK_NORM else nn.Identity()
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(D_MODEL, 2)
        q = q.view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        q, k = self.qn(q), self.kn(k)
        cos, sin = self.rot(T)
        q, k = apply_rope(q, k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).contiguous().view(B, T, C))

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        hid = ((int(8 * D_MODEL / 3) + 63) // 64) * 64
        self.w1 = nn.Linear(D_MODEL, hid, bias=False)
        self.w3 = nn.Linear(D_MODEL, hid, bias=False)
        self.w2 = nn.Linear(hid, D_MODEL, bias=False)
    def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = RMSNorm(D_MODEL), RMSNorm(D_MODEL)
        self.attn, self.mlp = Attn(), SwiGLU()
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

class MicroFriday(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.nf = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)
    def forward(self, idx, targets=None):
        x = self.tok(idx)
        for b in self.blocks: x = b(x)
        logits = self.head(self.nf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1),
                                   ignore_index=-100)
        return logits, loss

model = MicroFriday().to(DEVICE)
print(f"device={DEVICE}  params={sum(p.numel() for p in model.parameters()):,}")

decay = [p for p in model.parameters() if p.dim() >= 2]
nodecay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": WD}, {"params": nodecay, "weight_decay": 0.0}],
    lr=LR, betas=(0.9, 0.95), eps=1e-8)

def lr_at(s):
    if s < WARMUP: return LR * s / WARMUP
    p = (s - WARMUP) / max(1, STEPS - WARMUP)
    return 0.5 * LR * (1 + math.cos(math.pi * p))

MAX_NEW = (BLOCK - 12) if SCRATCHPAD else len(str(MAX_N * MAX_N)) + 3

@torch.no_grad()
def _generate_batch(problems):
    model.eval()
    prompts = [f"{a}{op}{b}=" for op, a, b in problems]
    groups = {}
    for i, pr in enumerate(prompts):
        groups.setdefault(len(pr), []).append(i)
    res = [""] * len(problems)
    for L, idx in groups.items():
        cur = torch.tensor([[stoi[c] for c in prompts[i]] for i in idx], device=DEVICE)
        done = torch.zeros(len(idx), dtype=torch.bool, device=DEVICE)
        for _ in range(MAX_NEW):
            logits, _ = model(cur)
            nxt = logits[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
            cur = torch.cat([cur, nxt[:, None]], 1)
            done |= nxt == EOS
            if bool(done.all()): break
        for row, i in zip(cur[:, L:].tolist(), idx):
            s = []
            for t in row:
                if t == EOS: break
                if t < len(chars): s.append(chars[t])
            res[i] = "".join(s)
    model.train()
    return res

def _generate(op, a, b):
    return _generate_batch([(op, a, b)])[0]

def _parse_int(s, rev=False):
    if not s: return None
    neg = s[0] == "-"; d = s[1:] if neg else s
    if not d or any(c not in "0123456789" for c in d): return None
    v = int(d[::-1] if rev else d)
    return -v if neg else v

def _answer(op, s):
    if SCRATCHPAD and op == "*":
        fin = [x for x in s.split(";") if x.startswith("=")]
        return _parse_int(fin[-1][1:]) if fin else None
    return _parse_int(s, rev=True)

def solve(op, a, b):
    return _answer(op, _generate(op, a, b))

def verbose(op, a, b):
    s = _generate(op, a, b)
    if not (SCRATCHPAD and op == "*"):
        return _answer(op, s), ""
    parts = []
    for seg in s.split(";"):
        if seg.startswith("="):
            parts.append(seg); continue
        if "=" not in seg:
            parts.append(seg + "‽"); continue
        L, R = seg.split("=", 1)
        try:
            if "*" in L: x, y = L.split("*"); ok = int(x) * int(y) == int(R)
            elif "+" in L: x, y = L.split("+"); ok = int(x) + int(y) == int(R)
            else: ok = False
        except Exception:
            ok = False
        parts.append(seg + ("✓" if ok else "✗"))
    return _answer(op, s), "  ".join(parts)

@torch.no_grad()
def acc(problems, n=150):
    if not problems: return 0.0
    s = random.sample(problems, min(n, len(problems)))
    outs = _generate_batch(s)
    return sum(_answer(p[0], o) == compute(*p) for p, o in zip(s, outs)) / len(s)

@torch.no_grad()
def val_loss(problems, batches=4):
    model.eval()
    t = sum(model(*make_batch(problems, BATCH_SIZE))[1].item() for _ in range(batches))
    model.train()
    return t / batches

PROBES = [random.choice(test_by_op[op]) for op in OPS if test_by_op[op]]
@torch.no_grad()
def show_probes():
    out = []
    for (op, a, b), o in zip(PROBES, _generate_batch(PROBES)):
        p = _answer(op, o); r = compute(op, a, b)
        out.append(f"{a}{op}{b}={r}→{p}{'✓' if p == r else '✗'}")
    return "  ".join(out)

hist = []
model.train()
print("\nлог: train_loss / val_loss / train_acc / TEST по операциям + живые примеры\n")
for step in tqdm(range(STEPS + 1)):
    for g in opt.param_groups: g["lr"] = lr_at(step)
    X, Y = make_batch(train_pool, BATCH_SIZE)
    _, loss = model(X, Y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % EVAL_EVERY == 0:
        vl = val_loss(test_all)
        tr = acc(train_pool)
        per = {op: acc(test_by_op[op]) for op in OPS}
        hist.append((step, loss.item(), vl, tr, per))
        per_s = "  ".join(f"{op}:{per[op]:.2f}" for op in OPS)
        tqdm.write(f"step {step:5d} | tl {loss.item():.3f} | vl {vl:.3f} | train {tr:.3f} | TEST[{per_s}]")
        tqdm.write(f"   {show_probes()}")

print("\nфинал на НЕВИДАННЫХ примерах (по 4 на операцию):")
for op in OPS:
    for op2, a, b in random.sample(test_by_op[op], min(4, len(test_by_op[op]))):
        r = compute(op2, a, b)
        if SCRATCHPAD and op2 == "*":
            p, steps = verbose(op2, a, b)
            print(f"  {a}*{b}={r}  модель:{p}  {'OK' if p == r else 'X'}")
            print(f"      шаги: {steps}")
        else:
            p = solve(op2, a, b)
            print(f"  {a:4d} {op2} {b:4d} = {r:5d}   модель: {str(p):>5}   {'OK' if p == r else 'X'}")

try:
    import matplotlib.pyplot as plt
    s = [h[0] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for op in OPS:
        ax[0].plot(s, [h[4][op] for h in hist], marker="o", label=f"TEST '{op}'")
    ax[0].plot(s, [h[3] for h in hist], "k--", alpha=0.5, label="train (все)")
    ax[0].set_ylim(-0.05, 1.05); ax[0].set_xlabel("шаг"); ax[0].set_ylabel("exact-match acc")
    ax[0].set_title("обобщение по операциям"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(s, [h[1] for h in hist], marker="o", label="train loss")
    ax[1].plot(s, [h[2] for h in hist], marker="o", label="val loss")
    ax[1].set_xlabel("шаг"); ax[1].set_ylabel("CE loss")
    ax[1].set_title("train vs val loss"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle(f"Арифметика {OPS}  MAX_N={MAX_N}  ALLOW_NEG={ALLOW_NEG}  TRAIN_POOL={len(train_pool)}")
    plt.tight_layout(); plt.show()
except Exception as e:
    print("график пропущен:", e)
