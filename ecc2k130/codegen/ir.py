# Straight-line program IR for generated bitsliced field arithmetic.
#
# A program is a DAG of word operations.  Every value is one machine word
# holding W independent bit-lanes (W walks in parallel), so an "op" here is one
# bit operation per lane and the op counts printed by the generator are directly
# comparable to the bit-operation counts in the ECC2K-130 papers.
#
# Passes, in the order recommended by the implementation guide:
#   1. build (with hash-consing, so duplicate subexpressions are shared)
#   2. LOP3 fusion   -- xor(xor(a,b),c) -> xor3, xor(a,and(b,c)) -> xorAnd
#   3. dead code elimination
#   4. slot allocation by liveness  (bounded number of C locals)
#   5. emission as C usable from both host and device code
#
# No type hints, camelCase identifiers, no itertools (project convention).

OP_XOR = 'xor'
OP_AND = 'and'
OP_XOR3 = 'xor3'
OP_XORAND = 'xorAnd'
OP_MAJ = 'maj'
OP_NOT = 'not'
OP_OR = 'or'

# LOP3 truth-table constants for a = 0xF0, b = 0xCC, c = 0xAA
LUT = {OP_XOR3: 0x96, OP_XORAND: 0x78, OP_MAJ: 0xE8}


class Prog:
    def __init__(self):
        self.ops = []          # index -> (op, args tuple)
        self.inputRef = []     # index -> (arrayName, position) or None
        self.hash = {}
        self.nInput = 0

    def addInput(self, arrayName, pos):
        idx = len(self.ops)
        self.ops.append(None)
        self.inputRef.append((arrayName, pos))
        self.nInput += 1
        return idx

    def _node(self, op, args):
        key = (op, args)
        got = self.hash.get(key)
        if got is not None:
            return got
        idx = len(self.ops)
        self.ops.append((op, args))
        self.inputRef.append(None)
        self.hash[key] = idx
        return idx

    def xor(self, a, b):
        if a is None:
            return b
        if b is None:
            return a
        if a == b:
            return None
        if a > b:
            a, b = b, a
        return self._node(OP_XOR, (a, b))

    def andOp(self, a, b):
        if a is None or b is None:
            return None
        if a == b:
            return a
        if a > b:
            a, b = b, a
        return self._node(OP_AND, (a, b))

    def notOp(self, a):
        if a is None:
            raise ValueError('not of constant zero needs an all-ones input')
        return self._node(OP_NOT, (a,))

    def orOp(self, a, b):
        if a is None:
            return b
        if b is None:
            return a
        if a == b:
            return a
        if a > b:
            a, b = b, a
        return self._node(OP_OR, (a, b))

    def xorList(self, items):
        # balanced tree keeps the critical path short and helps LOP3 fusion
        vals = []
        for v in items:
            if v is not None:
                vals.append(v)
        while len(vals) > 1:
            nxt = []
            i = 0
            while i + 1 < len(vals):
                nxt.append(self.xor(vals[i], vals[i + 1]))
                i += 2
            if i < len(vals):
                nxt.append(vals[i])
            vals = nxt
        return vals[0] if vals else None

    # ---- analysis -----------------------------------------------------
    def refCounts(self, roots):
        rc = [0] * len(self.ops)
        for r in roots:
            if r is not None:
                rc[r] += 1
        for i in range(len(self.ops) - 1, -1, -1):
            node = self.ops[i]
            if node is None or rc[i] == 0:
                continue
            for a in node[1]:
                rc[a] += 1
        return rc

    def liveNodes(self, roots):
        rc = self.refCounts(roots)
        out = []
        for i in range(len(self.ops)):
            if rc[i] > 0 or self.inputRef[i] is not None:
                out.append(i)
        return out

    def opCount(self, roots):
        rc = self.refCounts(roots)
        n = 0
        for i in range(len(self.ops)):
            if self.ops[i] is not None and rc[i] > 0:
                n += 1
        return n

    def instrCount(self, roots):
        """Instructions after LOP3 fusion (xor3/xorAnd/maj are one each)."""
        return self.opCount(roots)

    def bitOpCount(self, roots):
        """Cost in plain two-input bit operations, for comparison with the
        published bit-operation counts."""
        rc = self.refCounts(roots)
        n = 0
        for i in range(len(self.ops)):
            node = self.ops[i]
            if node is None or rc[i] == 0:
                continue
            op = node[0]
            if op == OP_XOR or op == OP_AND or op == OP_OR or op == OP_NOT:
                n += 1
            elif op == OP_XOR3 or op == OP_XORAND:
                n += 2
            elif op == OP_MAJ:
                n += 4
        return n

    # ---- LOP3 fusion --------------------------------------------------
    def fuseLop3(self, roots):
        rc = self.refCounts(roots)
        newOps = list(self.ops)
        for i in range(len(newOps)):
            node = newOps[i]
            if node is None or rc[i] == 0 or node[0] != OP_XOR:
                continue
            a, b = node[1]
            for x, y in ((a, b), (b, a)):
                ch = newOps[x]
                if ch is None or rc[x] != 1:
                    continue
                if ch[0] == OP_XOR:
                    newOps[i] = (OP_XOR3, (ch[1][0], ch[1][1], y))
                    rc[x] = 0
                    break
                if ch[0] == OP_AND:
                    newOps[i] = (OP_XORAND, (y, ch[1][0], ch[1][1]))
                    rc[x] = 0
                    break
        self.ops = newOps
        self.hash = {}
        return self

    # ---- emission -----------------------------------------------------
    def emit(self, roots, outName, indent='    ', wordType='W'):
        """Return a list of C source lines computing roots into outName[k]."""
        rc = self.refCounts(roots)
        order = []
        for i in range(len(self.ops)):
            if self.ops[i] is not None and rc[i] > 0:
                order.append(i)

        rootsOf = {}
        for k in range(len(roots)):
            if roots[k] is not None:
                rootsOf.setdefault(roots[k], []).append(k)
        lastUse = {}
        for i in order:
            for a in self.ops[i][1]:
                lastUse[a] = i

        slotOf = {}
        free = []
        nextSlot = [0]

        def alloc(node):
            if free:
                s = free.pop()
            else:
                s = nextSlot[0]
                nextSlot[0] += 1
            slotOf[node] = s
            return s

        def ref(node):
            src = self.inputRef[node]
            if src is not None:
                return '%s[%d]' % (src[0], src[1])
            return 't%d' % slotOf[node]

        lines = []
        for i in order:
            op, args = self.ops[i]
            texts = []
            for a in args:
                texts.append(ref(a))
            for a in args:
                if self.inputRef[a] is None and lastUse.get(a) == i and a in slotOf:
                    free.append(slotOf[a])
                    del slotOf[a]
            s = alloc(i)
            if op == OP_XOR:
                expr = '%s ^ %s' % (texts[0], texts[1])
            elif op == OP_AND:
                expr = '%s & %s' % (texts[0], texts[1])
            elif op == OP_XOR3:
                expr = 'ECC_XOR3(%s, %s, %s)' % (texts[0], texts[1], texts[2])
            elif op == OP_XORAND:
                expr = 'ECC_XORAND(%s, %s, %s)' % (texts[0], texts[1], texts[2])
            elif op == OP_MAJ:
                expr = 'ECC_MAJ(%s, %s, %s)' % (texts[0], texts[1], texts[2])
            elif op == OP_OR:
                expr = '%s | %s' % (texts[0], texts[1])
            elif op == OP_NOT:
                expr = '~%s' % texts[0]
            else:
                raise ValueError('unknown op ' + op)
            lines.append('%st%d = %s;' % (indent, s, expr))
            if i in rootsOf:
                for k in rootsOf[i]:
                    lines.append('%s%s[%d] = t%d;' % (indent, outName, k, s))
                if lastUse.get(i) is None:
                    free.append(s)
                    del slotOf[i]

        for k in range(len(roots)):
            r = roots[k]
            if r is None:
                lines.append('%s%s[%d] = ECC_ZERO;' % (indent, outName, k))
            elif self.inputRef[r] is not None:
                lines.append('%s%s[%d] = %s;' % (indent, outName, k, ref(r)))
        decl = ''
        if nextSlot[0]:
            names = []
            for s in range(nextSlot[0]):
                names.append('t%d' % s)
            decl = '%s%s %s;' % (indent, wordType, ', '.join(names))
        return ([decl] if decl else []) + lines, nextSlot[0]

    # ---- verification -------------------------------------------------
    def evaluate(self, inputValues, roots):
        """Bit-parallel evaluation: inputValues maps (array, pos) -> int."""
        val = [0] * len(self.ops)
        for i in range(len(self.ops)):
            src = self.inputRef[i]
            if src is not None:
                val[i] = inputValues.get(src, 0)
                continue
            node = self.ops[i]
            if node is None:
                continue
            op, args = node
            if op == OP_XOR:
                val[i] = val[args[0]] ^ val[args[1]]
            elif op == OP_AND:
                val[i] = val[args[0]] & val[args[1]]
            elif op == OP_XOR3:
                val[i] = val[args[0]] ^ val[args[1]] ^ val[args[2]]
            elif op == OP_XORAND:
                val[i] = val[args[0]] ^ (val[args[1]] & val[args[2]])
            elif op == OP_OR:
                val[i] = val[args[0]] | val[args[1]]
            elif op == OP_NOT:
                val[i] = ~val[args[0]]
            elif op == OP_MAJ:
                a, b, c = val[args[0]], val[args[1]], val[args[2]]
                val[i] = (a & b) | (a & c) | (b & c)
        out = []
        for r in roots:
            out.append(0 if r is None else val[r])
        return out
