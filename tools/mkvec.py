import random, sys
def ref(a, b, op):
    if op == 0: return b, 0
    if op == 1: return a ^ b, 0
    if op == 2: return a & b, 0
    if op == 3: return a | b, 0
    if op == 4: s = a + b; return s & 255, s >> 8
    if op == 5: s = a + ((~b) & 255) + 1; return s & 255, s >> 8
    if op == 6: return b >> 1, b & 1
    if op == 7: return ((a & 15) * (b & 15)) & 255, 0
mode = sys.argv[1]
print("A[8] B[8] Opcode[3] O[8] Cout")
b8 = lambda v: format(v, '08b')
if mode == 'all':
    rows = [(a, b, op) for op in range(8) for a in range(256) for b in range(256)]
else:
    random.seed(1)
    edge = [0, 1, 2, 15, 16, 127, 128, 254, 255]
    rows = [(a, b, op) for op in range(8) for a in edge for b in edge]
    rows += [(random.randrange(256), random.randrange(256), random.randrange(8)) for _ in range(3000)]
for a, b, op in rows:
    o, c = ref(a, b, op)
    print(b8(a), b8(b), format(op, '03b'), b8(o), c)
