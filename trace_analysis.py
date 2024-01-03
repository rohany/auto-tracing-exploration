#!/usr/bin/env python3

import argparse
from collections import deque
import copy
import ctypes
import itertools
import os
import re
import time

from enum import Enum
from suffix_tree import Tree
from suffix_tree.node import Node, Internal, Leaf

from typing import Sequence, Hashable

prefix = "\[(?P<node>[0-9]+) - (?P<thread>[0-9a-f]+)\](?:\s+[0-9]+\.[0-9]+)? \{\w+\}\{legion_spy\}: "
index_task_pat = re.compile(
    prefix+"Index Task (?P<ctx>[0-9]+) (?P<tid>[0-9]+) (?P<uid>[0-9]+) "+
    "(?P<index>[0-9]+) (?P<name>.+)")
requirement_pat = re.compile(
    prefix+"Logical Requirement (?P<uid>[0-9]+) (?P<index>[0-9]+) (?P<is_reg>[0-1]) "+
    "(?P<ispace>[0-9a-f]+) (?P<fspace>[0-9]+) (?P<tid>[0-9]+) (?P<priv>[0-9]+) "+
    "(?P<coher>[0-9]+) (?P<redop>[0-9]+) (?P<pis>[0-9a-f]+)")
req_field_pat           = re.compile(
    prefix+"Logical Requirement Field (?P<uid>[0-9]+) (?P<index>[0-9]+) (?P<fid>[0-9]+)")
req_proj_pat            = re.compile(
    prefix+"Logical Requirement Projection (?P<uid>[0-9]+) (?P<index>[0-9]+) "+
    "(?P<pid>[0-9]+)")
op_provenance_pat        = re.compile(
    prefix+"Operation Provenance (?P<uid>[0-9]+) (?P<provenance>.*)")


TRACE_MIN_LENGTH = 3

class LegionObject:
    def __init__(self, uid):
        self.uid = uid

    def __str__(self):
        return f"{type(self).__name__}({self.uid})"

    def __hash__(self):
        return hash(self.uid)


class FieldSpace(LegionObject):
    ...


class IndexSpace(LegionObject):
    ...


class IndexPartition(LegionObject):
    ...


class ProjectionFunction(LegionObject):
    ...


class Partition:
    def __init__(self, iid, fid, tid):
        self.iid = iid
        self.fid = fid
        self.tid = tid

    def __str__(self):
        return f"Partition({self.iid}, {self.fid}, {self.tid})"

    def __hash__(self):
        return hash((self.iid, self.fid, self.tid))


class Region:
    def __init__(self, iid, fid, tid):
        self.iid = iid
        self.fid = fid
        self.tid = tid

    def __str__(self):
        return f"Region({self.iid}, {self.fid}, {self.tid})"

    def __hash__(self):
        return hash((self.iid, self.fid, self.tid))



class Requirement:
    def __init__(self, state, index, is_reg, index_node, field_space, tid, logical_node, priv, coher, redop, parent):
        self.state = state
        self.index = index
        self.is_reg = is_reg
        self.index_node = index_node
        self.field_space = field_space
        self.tid = tid
        self.logical_node = logical_node
        self.priv = priv
        self.coher = coher
        self.redop = redop
        self.parent = parent
        self.fields = list()
        self.projection_function = None

    def add_field(self, fid):
        self.fields.append(fid)

    def set_projection_function(self, proj_func):
        if self.projection_function:
            assert proj_func is self.projection_function
        else:
            self.projection_function = proj_func

    def __str__(self):
        if self.projection_function.uid != 0:
            return f"Req(logical_node={self.logical_node}, fields={self.fields}, proj={self.projection_function})"
        else:
            return f"Req(logical_node={self.logical_node}, fields={self.fields})"

    def __hash__(self):
        return hash((
            self.index,
            self.is_reg,
            self.index_node,
            self.field_space,
            self.tid,
            self.logical_node,
            self.priv,
            self.coher,
            self.redop,
            self.parent,
            tuple(self.fields),
            self.projection_function,
        ))


class OperationKind(Enum):
    INDEX_TASK = 0


class Operation:
    def __init__(self, uid):
        self.uid = uid
        self.kind = None
        self.taskid = None
        self.reqs = dict()
        self.name = None
        self.provenance = None

    def set_task_id(self, taskid):
        self.taskid = taskid

    def set_kind(self, kind):
        self.kind = kind

    def set_name(self, name):
        self.name = name

    def add_requirement(self, requirement):
        assert requirement.index not in self.reqs
        self.reqs[requirement.index] = requirement

    def add_requirement_field(self, index, fid):
        assert index in self.reqs
        self.reqs[index].add_field(fid)

    def set_projection_function(self, index, proj_func):
        assert index in self.reqs
        self.reqs[index].set_projection_function(proj_func)

    def __str__(self):
        assert self.kind == OperationKind.INDEX_TASK
        build = f"IndexTask(name={self.name}, requirements={{"
        for i, (_, req) in enumerate(list(sorted(self.reqs.items()))):
            if i > 0:
                build += ", "
            build += str(req)
        build += "}})"
        return build

    def __repr__(self):
        return str(self)

    def __hash__(self):
        assert self.kind == OperationKind.INDEX_TASK
        return hash((
            self.kind,
            self.taskid,
            tuple(self.reqs.values()),
        ))

class State:
    def __init__(self):
        self.ops = dict()
        self.prog = []
        self.field_spaces = dict()
        self.projection_functions = dict()
        self.regions = dict()
        self.index_spaces = dict()
        self.index_partitions = dict()
        self.partitions = dict()

    def prune_ops(self):
        self.prog = [op for op in self.prog if op.kind == OperationKind.INDEX_TASK]

    def get_operation(self, uid, kind=None):
        if uid in self.ops:
            if kind != self.ops[uid].kind:
                return None
            return self.ops[uid]
        if kind is not None:
            return None
        result = Operation(uid)
        self.ops[uid] = result
        self.prog.append(result)
        return result

    def get_field_space(self, fid):
        if fid in self.field_spaces:
            return self.field_spaces[fid]
        result = FieldSpace(fid)
        self.field_spaces[fid] = result
        return result

    def get_projection_function(self, pid):
        if pid in self.projection_functions:
            return self.projection_functions[pid]
        result = ProjectionFunction(pid)
        self.projection_functions[pid] = result
        return result

    def get_region(self, iid, fid, tid):
        key = (iid, fid, tid)
        if key in self.regions:
            return self.regions[key]
        result = Region(iid, fid, tid)
        self.regions[key] = result
        return result

    def get_index_space(self, uid):
        if uid in self.index_spaces:
            return self.index_spaces[uid]
        result = IndexSpace(uid)
        self.index_spaces[uid] = result
        return result

    def get_index_partition(self, pid):
        if pid in self.index_partitions:
            return self.index_partitions[pid]
        result = IndexPartition(pid)
        self.index_partitions[pid] = result
        return result

    def get_partition(self, iid, fid, tid):
        key = (iid, fid, tid)
        if key in self.partitions:
            return self.partitions[key]
        result = Partition(iid, fid, tid)
        self.partitions[key] = result
        return result

def parse_spy_log(file):
    state = State()
    for lineno, line in enumerate(file.readlines()):
        m = index_task_pat.match(line)
        if m is not None:
            op = state.get_operation(int(m.group('uid')))
            op.set_kind(OperationKind.INDEX_TASK)
            op.set_name(m.group('name'))
            op.set_task_id(int(m.group('tid')))
            continue
        # TODO (rohany): Handle single task launches...
        m = requirement_pat.match(line)
        if m is not None:
            op = state.get_operation(int(m.group('uid')), kind=OperationKind.INDEX_TASK)
            if not op:
                continue
            is_reg = True if int(m.group('is_reg')) == 1 else False
            field_space = state.get_field_space(int(m.group('fspace')))
            tid = int(m.group('tid'))
            priv = int(m.group('priv'))
            coher = int(m.group('coher'))
            redop = int(m.group('redop'))
            parent = state.get_region(int(m.group('pis'),16), field_space.uid, tid)
            if is_reg:
                index_space = state.get_index_space(int(m.group('ispace'),16))
                region = state.get_region(index_space.uid, field_space.uid, tid)
                requirement = Requirement(state, int(m.group('index')), True,
                                          index_space, field_space, tid, region,
                                          priv, coher, redop, parent)
                op.add_requirement(requirement)
            else:
                index_partition = state.get_index_partition(int(m.group('ispace'),16))
                partition = state.get_partition(index_partition.uid, field_space.uid, tid)
                requirement = Requirement(state, int(m.group('index')), False,
                                          index_partition, field_space, tid, partition,
                                          priv, coher, redop, parent)
                op.add_requirement(requirement)
            continue
        m = req_field_pat.match(line)
        if m is not None:
            op = state.get_operation(int(m.group('uid')), kind=OperationKind.INDEX_TASK)
            if not op:
                continue
            index = int(m.group('index'))
            fid = int(m.group('fid'))
            op.add_requirement_field(index, fid)
            continue
        m = req_proj_pat.match(line)
        if m is not None:
            op = state.get_operation(int(m.group('uid')), kind=OperationKind.INDEX_TASK)
            if not op:
                continue
            index = int(m.group('index'))
            func = state.get_projection_function(int(m.group('pid')))
            op.set_projection_function(index, func)
            continue
        m = op_provenance_pat.match(line)
        if m is not None:
            op = state.get_operation(int(m.group('uid')))
            op.provenance = m.group('provenance')
            continue
    return state

from collections import defaultdict

# Suffix array construction algorithm taken from http://www.cs.cmu.edu/~15451-f20/LectureNotes/lec25-suffarray.pdf,
# and translated to Python from OCaml.
def suffix_array(s : Sequence[Hashable]):
    # TODO (rohany): This implementation only works when s is an array of non-negative integers. Though it can work
    #  for hashes if we just use unsigned integers? I think the non-negative part only comes from using -1 as the sentinel.
    #  Which again, we can't do if we're going to use hashes.
    # Either we cap all hashes to 63 bits and then use signed integers for the sort, or use unsigned integers
    # and error out if we find a hash value equal to 0 or something. I don't want to box all of my integers
    # in the sort into `int option` types, but we can do it if we have to.
    small = -1
    n = len(s)
    if type(s) is str:
        a = [ord(c) for c in s]
    else:
        a = [c for c in s]
    w = [((0, 0), 0)] * n
    def sortpass(shift):
        geta = lambda j : a[j] if j < n else small
        for i in range(0, n):
            w[i] = ((a[i], geta(i + shift)), i)
        w.sort()
        a[w[0][1]] = 0
        for j in range(1, n):
            # Unsure a little bit about this, but we'll see...
            a[w[j][1]] = a[w[j-1][1]] + (0 if w[j-1][0] == w[j][0] else 1)
        if a[w[n-1][1]] < n-1:
            sortpass(shift * 2)
    sortpass(1)
    return [t[1] for t in w]

# find all longest common prefixes in string s
def lcp_array(s, sa):
    # Kasai et al's algorithm:
    # (https://gist.github.com/prasoon2211/cc3f3d5b43a0885c0e7a)
    n = len(s)
    k = 0
    lcp = [0] * n
    rank = [0] * n
    for i in range(n):
        rank[sa[i]] = i
    for i in range(n):
        if rank[i] == n-1:
            k = 0
            continue
        j = sa[rank[i] + 1]
        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1
        lcp[rank[i]] = k
        if k:
            k -= 1
    return lcp

# longest repeating substring
def lrs(s, sa, lcp):
    max_lcp = max(lcp)
    max_index = max(range(len(lcp)), key=lcp.__getitem__)
    return s[sa[max_index] : sa[max_index] + max_lcp]

def maximal_repeats(s, sa, lcp):
    maxrep = defaultdict(int)
    for i in range(len(s)):
        maxrep[i] = len(s) + 1
    for i in range(len(s)-1):
        p = max(lcp[i], lcp[i+1])
        maxrep[sa[i] + p - 1] = min(maxrep[sa[i] + p - 1], sa[i])
    return maxrep

# TODO (rohany): This is an n^2, maybe even n^3 algorithm???
# given: a token stream (s), suffix array of s (sa), and list of longest common prefixes in s (lcp),
# return: dictionary of tandem repeats in s, mapping repeated token sequence to pair of (start index, number of repeats)
def tandem_repeats(s, sa, lcp):
    tandem_repeats = {}
    for i, suffix_index in enumerate(sa):
        # repeat = s[suffix_index:suffix_index + lcp[i]]
        # TODO (rohany): This version is necessary to do when a "string" is a list of integers.
        repeat = s[suffix_index:suffix_index + lcp[i]]
        if not repeat:
            continue
        repeat = tuple(repeat)
        for j in range(0, len(sa)):
            suffix = s[sa[j]:]
            repeats = 1
            try:
                repeat_start = suffix.index(repeat[0])
            except ValueError:
                continue
            while True:
                # Again this needs to be tuple here when "string" is a list of integers.
                if repeat * repeats != tuple(suffix[repeat_start:repeat_start + len(repeat) * repeats]):
                    break
                repeats += 1
            repeats -= 1
            if repeats > 1:
                if repeat in tandem_repeats and tandem_repeats[repeat][1] > repeats:
                    continue
                tandem_repeats[repeat] = (sa[j] + repeat_start, repeats)
    return tandem_repeats

# This is really fast but doesn't seem to give all of the same results that the first
# implementation gives ...
def another_tandem_repeat(s, sa, lcp):
    bestlen, results = 0, []
    n = len(s)
    for sai in range(n-1):
        i = sa[sai]
        c = n
        for saj in range(sai + 1, n):
            c = min(c, lcp[saj])
            if not c:
                break
            j = sa[saj]
            w = abs(i - j)
            if c < w:
                continue
            numreps = 1 + c // w
            assert numreps > 1
            total = w * numreps
            if total >= bestlen:
                if total > bestlen:
                    results.clear()
                    bestlen = total
                results.append((min(i, j), w, numreps))
    return bestlen, results


def naivetrepeats(s):
    from collections import deque

    # There are zcount equal characters starting
    # at index starti.
    def update(starti, zcount):
        nonlocal bestlen
        while zcount >= width:
            numreps = 1 + zcount // width
            count = width * numreps
            if count >= bestlen:
                if count > bestlen:
                    results.clear()
                results.append((starti, width, numreps))
                bestlen = count
            else:
                break
            zcount -= 1
            starti += 1

    bestlen, results = 0, []
    t = deque(s)
    for width in range(1, len(s) // 2 + 1):
        t.popleft()
        zcount = 0
        for i, (a, b) in enumerate(zip(s, t)):
            if a == b:
                if not zcount: # new run starts here
                    starti = i
                zcount += 1
            # else a != b, so equal run (if any) ended
            elif zcount:
                update(starti, zcount)
                zcount = 0
        if zcount:
            update(starti, zcount)
    return bestlen, results


def longest_nonoverlapping_repeats(program):
    # Potential suffix tree implementation for C++ -- https://github.com/murraycu/murrayc-suffix-tree.
    tree = Tree({"prog": [hash(op) for op in program]})

    # First, for each node in the suffix tree, calculate all of
    # the leaves that are present in the subtree rooted at that node.
    def collect_children(node):
        if isinstance(node, Leaf):
            node.all_subtrees = [node]
        else:
            all_subtrees = []
            for _, child in node.children.items():
                all_subtrees.extend(child.all_subtrees)
            node.all_subtrees = all_subtrees
    tree.post_order(collect_children)

    # https://stackoverflow.com/a/57650024. At each node, record the minimum
    # starting position and maximum starting position of suffixes rooted
    # at this node. This can be used to tell us if the substring represented
    # at this node is repeated in a non-overlapping way, i.e. if there exists
    # a substring that that also starts with this node but is at least
    # len(node) characters away.
    def minmax(node):
        if isinstance(node, Leaf):
            node.imin = node.start
            node.imax = node.start
        else:
            assert(isinstance(node, Internal))
            node.imin = min([child.imin for _, child in node.children.items()])
            node.imax = max([child.imax for _, child in node.children.items()])
    tree.post_order(minmax)

    # Finally, collect all nodes that satisfy the condition described above. We
    # also augment the traversal to discover how many repeats each substring
    # actually has.
    repeats = []
    def finder(node):
        if isinstance(node, Internal) and node != tree.root:
            strlen = node.end - node.start
            if node.imin + strlen <= node.imax:
                repeats.append(node)
                # For each of the leaves rooted at this subtree, figure out how many
                # of them are at least strlen places apart.
                count = 0
                sorted_children = list(sorted(node.all_subtrees, key=lambda x: x.start))
                current = sorted_children[0]
                for i in range(1, len(sorted_children)):
                    next = sorted_children[i]
                    if current.start + strlen <= next.start:
                        current = next
                        count += 1
                node.num_repeats = count + 1
    tree.pre_order(finder)

    # This ordering heuristic is approximating tandem repeats?
    return list(reversed(sorted(repeats, key=lambda x: ((x.end - x.start) * x.num_repeats, x.num_repeats))))


class RepeatAlgorithms(Enum):
    LONGEST_NONOVERLAPPING_REPEAT = 0
    TANDEM_REPEATS_LOOP_REROLL = 1
    TANDEM_REPEATS_NAIVE = 2
    TANDEM_REPEATS_LIMITED = 3


def find_repeat_candidates(program, alg):
    if alg == RepeatAlgorithms.LONGEST_NONOVERLAPPING_REPEAT:
        return longest_nonoverlapping_repeats(program)
    elif alg == RepeatAlgorithms.TANDEM_REPEATS_NAIVE:
        return naivetrepeats([hash(op) for op in program])
    elif alg in (RepeatAlgorithms.TANDEM_REPEATS_LOOP_REROLL, RepeatAlgorithms.TANDEM_REPEATS_LOOP_REROLL):
        if type(program) is str:
            S = program
        else:
            S = [int(ctypes.c_size_t(hash(op)).value) for op in program]
        sarray = suffix_array(S)
        lcp = lcp_array(S, sarray)
        if alg == RepeatAlgorithms.TANDEM_REPEATS_LOOP_REROLL:
            return tandem_repeats(S, sarray, lcp)
        elif alg == RepeatAlgorithms.TANDEM_REPEATS_LIMITED:
            return another_tandem_repeat(S, sarray, lcp)
    else:
        assert(False)


class TraceProcessor:
    # TODO (rohany): Adjust the signature here.
    def process_operation(self, op: Operation):
        assert(False)


# TODO (rohany): Add in a tiered trace identification algorithm.
class BatchedTraceProcessor(TraceProcessor):
    def __init__(self, prog, batchsize):
        self.batch = []
        self.batchsize = batchsize
        self.hashed_prog = [hash(op) for op in prog]
        self.opidx = -1

    def reset(self):
        self.batch = []

    def process_operation(self, op: Operation):
        self.opidx += 1
        self.batch.append(op)
        if len(self.batch) == self.batchsize:
            # TODO (rohany): Adjust this analysis to return some set of potential potential traces
            #  that don't overlap with each other. This is pretty critical, as we can't decide
            #  to record a trace while already recording one!
            repeats = find_repeat_candidates(self.batch, RepeatAlgorithms.LONGEST_NONOVERLAPPING_REPEAT)

            def startswith(l1, l2):
                for i1, i2 in zip(l1, l2):
                    if i1 != i2:
                        return False
                return True

            def anyprefix(l, ls):
                for l2 in ls:
                    if startswith(l, l2):
                        return True
                return False

            def hashall(l):
                return [hash(x) for x in l]

            # TODO (rohany): I think that this isn't the right place for this / the better
            #  place to do this is inside the trie itself. Don't accept insertions that are
            #  prefixes of anything else already in the trace. However, this is fine for now,
            #  as it seems to be working.
            # TODO (rohany): Report the top N traces that don't overlap with each other.
            # TODO (rohany): I don't know yet if the overlapping business is the
            #  right thing or not?
            to_return = []
            for repeat in repeats:
                trace = self.batch[repeat.start:repeat.end]
                to_return.append(trace)

            # if len(repeats) != 0:
            #     best = repeats[0]
            #     S = self.hashed_prog
            #     start, end = best.start, best.end
            #     trace = tuple([hash(op) for op in self.batch][start:end])
            #     count = 0
            #     lastmatch = None
            #     for j in range(self.opidx, len(S)):
            #         if trace == tuple(S[j:j+(end-start)]) and (lastmatch is None or (j - lastmatch >= len(trace))):
            #             count += 1
            #             lastmatch = j
            #     print(f"Found repeat at operation index: {self.opidx}: repeatlen={best.end-best.start}, num_repeats={best.num_repeats}, full total matches={count}")

            # TODO (rohany): This should be a circular buffer, rather than a "cleared" buffer?
            self.batch = []
            return to_return
        return None


class RollingKGramHasher:
    class HasherState(Enum):
        INIT = 0
        FULL = 1

    def __init__(self, k):
        self.k = k
        # "base" = large random prime
        self.b = 2452301417224347413
        # The modulus is 2^64, as we'll let integers overflow. We have to
        # manually do this in Python, but will overflow by default in C++.
        self.p = 18446744073709551615
        self.bk = 1
        for i in range(1, k):
            self.bk = (self.bk * self.b) % self.p
        self.state = self.HasherState.INIT
        self.curhash = 0
        self.buf = [0] * self.k
        self.idx = 0
        self.seen = 0

    def reset(self):
        self.state = self.HasherState.INIT
        self.curhash = 0
        self.idx = 0
        self.seen = 0

    def add_hash(self, hashval):
        if self.state == self.HasherState.INIT:
            self.buf[self.idx] = hashval
            self.idx += 1
            old = (self.curhash * self.b) % self.p
            self.curhash = (old + hashval) % self.p
            self.seen += 1
            if self.seen == self.k:
                self.state = self.HasherState.FULL
                return self.curhash
            return None
        elif self.state == self.HasherState.FULL:
            idx = self.idx % self.k
            self.idx += 1
            oldtok = self.buf[idx]
            self.buf[idx] = hashval
            sub = (oldtok * self.bk) % self.p
            diff = (self.curhash - sub) % self.p
            mul = (diff * self.b) % self.p
            result = (mul + hashval) % self.p
            self.curhash = result
            return result
        else:
            assert(False)


# Implementation taken from Figure 5 of http://theory.stanford.edu/~aiken/publications/papers/sigmod03.pdf.
class Winnower:
    def __init__(self, t, k):
        self.t = t
        self.k = k
        self.w = t - k + 1
        assert(self.w > 0)
        # Circular buffer holding the windows.
        self.h = [None] * self.w
        self.r = 0
        self.min = 0

        self.hasher = RollingKGramHasher(k)
        self.watches = {}

        # TODO (rohany): Make this configurable.
        self.threshold = 5

        # Keeping the below thoughts around just for logging, but I don't
        # think they are correct. We don't need to explicitly maintain
        # the winnowed fingerprints themselves, all we need to maintain is
        # either a hash table or a sketch of the fingerprints seen. In a different
        # component we're actually going to maintain the hashes and the tasks
        # themselves, so the purpose of this winnower is to just maintain an
        # approximate count of each fingerprint, and report when a fingerprint
        # has crossed a threshold. When it has crossed a threshold, we reset
        # all of the counting state within the winnower, and notify the larger batch processing
        # algorithm that it is time to find actual repeats.

        # WRONG BELOW:
        # How many fingerprints to track. In a standard application of
        # winnowing, we would collect a set of fingerprints for the entire
        # document. Since we are more interested in a stream
        # self.fingerlen = fingerlen

    def minfunc(self, a, b):
        return a if self.lessfunc(a, b) else b

    def lessfunc(self, a, b):
        if a is None and b is None:
            return False
        if a is None and b is not None:
            return True
        if a is not None and b is None:
            return False
        return a < b

    # TODO (rohany): I'm not sure what the overall interface will look
    #  like here, as different "process_operation" objects are going
    #  to want to return different kinds of data naturally, but for now
    #  all this needs to do is to return to the caller whether the
    #  larger buffer analysis needs to be done.
    def process_operation(self, op: Operation) -> bool:
        kgramhash = self.hasher.add_hash(hash(op))
        # print(op, "KGRAMHASH:", kgramhash, self.hasher.buf)
        if kgramhash is not None:
            return self.process_hash(kgramhash)
        return False

    def process_hash(self, hashval):
        self.r = (self.r + 1) % self.w
        self.h[self.r] = hashval
        if self.min == self.r:
            i = (self.r - 1) % self.w
            while i != self.r:
                if self.lessfunc(self.h[i], self.h[self.min]):
                    self.min = i
                i = (i - 1 + self.w) % self.w
            return self.record(self.h[self.min])
        else:
            # Right now, this implements what the authors call "robust winnowing". If
            # we want "normal winnowing", then this needs to be a <=, rather than a <.
            if self.lessfunc(self.h[self.r], self.h[self.min]):
                self.min = self.r
                return self.record(self.h[self.min])
        return False

    def record(self, hashval) -> bool:
        # For now, or for a first implementation, we can just maintain a hash
        # table of each fingerprint and its counts. The problem is that this
        # data structure could increase to be the total number of elements in
        # the stream. To handle that case (and use less memory), we could instead
        # implement a streaming style algorithm (in particular the "lossy counting"
        # algorithm described in https://www.vldb.org/conf/2002/S10P03.pdf, and
        # explained better in https://www.cs.emory.edu/~cheung/Courses/584/Syllabus/07-Heavy/Manku.html.
        count = self.watches.get(hashval, 0)
        count += 1
        self.watches[hashval] = count
        return count > self.threshold

    def reset(self):
        self.watches = {}
        self.r = 0
        self.min = 0
        self.hasher.reset()


class WinnowingTraceProcessor(TraceProcessor):
    class State(Enum):
        BUF_INIT = 0
        BUF_FULL = 1

    def __init__(self, batchsize):
        self.batch = [None] * batchsize
        self.idx = 0
        self.state = self.State.BUF_INIT
        self.batchsize = batchsize
        # TODO (rohany): play with these parameters later.
        self.winnower = Winnower(10, 5)

    def add_op(self, op: Operation):
        self.batch[self.idx % self.batchsize] = op
        self.idx += 1
        if self.state == self.State.BUF_INIT and self.idx == self.batchsize:
            self.state = self.State.BUF_FULL

    def export_buf(self):
        if self.state == self.State.BUF_INIT:
            return self.batch[:self.idx]
        else:
            result = [None] * self.batchsize
            for i in range(0, self.batchsize):
                result[i] = self.batch[(self.idx + i) % self.batchsize]
            return result

    def process_operation(self, op: Operation):
        self.add_op(op)
        trigger = self.winnower.process_operation(op)
        # print("TRIGGER VALUE: ", trigger)
        # TODO (rohany): I'm not sure how to incorporate both the buffer and the
        #  winnower here. I think something to do (in a separate class) is to have
        #  a batched + triggered processor that starts a processing job if the trigger
        #  is hit or if the buffer size is reached. In the case that the buffer size
        #  has been reached, flush the buffer away.
        if trigger:
            self.winnower.reset()
            batch = self.export_buf()
            repeats = find_repeat_candidates(batch, RepeatAlgorithms.LONGEST_NONOVERLAPPING_REPEAT)
            to_return = []
            for repeat in repeats:
                trace = batch[repeat.start:repeat.end]
                to_return.append(trace)
            return to_return
        return None


class WinnowingBatchedTraceProcessor(TraceProcessor):
    def __init__(self, batchsize):
        self.batch = []
        self.batchsize = batchsize
        # TODO (rohany): play with these parameters later.
        self.winnower = Winnower(10, 5)

    def process_operation(self, op: Operation):
        self.batch.append(op)
        trigger = self.winnower.process_operation(op)
        if trigger or len(self.batch) == self.batchsize:
            batch = self.batch
            self.winnower.reset()
            self.batch = []
            repeats = find_repeat_candidates(batch, RepeatAlgorithms.LONGEST_NONOVERLAPPING_REPEAT)
            to_return = []
            for repeat in repeats:
                trace = batch[repeat.start:repeat.end]
                to_return.append(trace)
            return to_return
        return None

    def reset(self):
        self.batch = []
        self.winnower.reset()


# Another component of the trace cache architecture is a component that actually
# sees when repeated substrings are occurring before it tries to memoize them.
# This component needs to:
# 1) maintain a set of current "potential" traces, i.e. traces returned from
#    the trace identification component.
# 2) as operations are being processed, maintain counts of what traces are
#    actually getting hit by the program.
# 3) after some number of operations have been processed, or a trace in the
#    counter has reached a threshold value, report the counts of all traces
#    that are being maintained.
# 4) The traces identified by the repeat-based algorithm and then verified
#    by this component can now be recorded if seen again.
# This component could be implemented in the following manner:
# 1) Maintain a trie built out of all potential traces T_1, ... T_n. In the real
#    implementation, we can use a radix tree.
# 2) Maintain a set of "active trie pointers".
# 3) As each operation is encountered, advance all potential trie pointers
#    if possible. Remove any that are not possible to advance. Add in a pointer
#    for the current operation.
# 4) If a pointer ever makes it to the bottom of the trie, increment the pointer
#    for that string.

class TrieNode:
    def __init__(self, token):
        self.token = token
        self.is_end = False
        self.children = {}
        self.num_visits = 0
        self.opidx = None

    def print(self):
        print("Parent: ", self, self.token, self.children)
        for child in self.children.values():
            child.print()

    def empty(self) -> bool:
        return len(self.children) == 0


class Trie:
    def __init__(self):
        self.root = TrieNode(None)

    def insert(self, string, opidx):
        node = self.root

        for token in string:
            if token in node.children:
                node = node.children[token]
            else:
                new_node = TrieNode(token)
                node.children[token] = new_node
                node = new_node

        assert node.opidx is None
        assert not node.is_end
        node.is_end = True
        node.opidx = opidx

        # self.check_invariants()

    def prefix(self, string):
        node = self.root

        for token in string:
            if token in node.children:
                node = node.children[token]
            else:
                return False

        return True

    def superstring(self, string):
        node = self.root

        index = 0
        while index < len(string):
            token = string[index]
            if token in node.children:
                node = node.children[token]
            else:
                break
            index += 1

        if node == self.root:
            return False

        return (index < len(string)) and node.is_end

    def contains(self, string):
        node = self.root

        for token in string:
            if token in node.children:
                node = node.children[token]
            else:
                return False

        return node.is_end

    def remove(self, string):
        def recur(node, idx):
            if idx == len(string):
                node.is_end = False
                if node.empty():
                    return None
                else:
                    return node
            else:
                # TODO (rohany): Let's just assume that we'll only call
                #  delete on strings that were actually inserted into the trie.
                token = string[idx]
                child = recur(node.children[token], idx+1)
                if child is None:
                    del node.children[token]
                else:
                    node.children[token] = child
                if not node.is_end and node.empty():
                    return None
                else:
                    return node
        recur(self.root, 0)

    def foreach_string(self, f):
        def recur(node, prefix):
            local = prefix.copy()
            if node.token is not None:
                local.append(node.token)
            if node.is_end:
                f(local, node)
            for child in node.children.values():
                recur(child, local)

        recur(self.root, [])

    def print(self):
        self.root.print()

    def check_invariants(self):
        # Only nodes that are ends of strings can have opidx set.
        def endopidx(node):
            if node.is_end and node.opidx is None:
                assert False
            if not node.is_end and node.opidx is not None:
                assert False
            for child in node.children.values():
                endopidx(child)
        endopidx(self.root)
        # ...


class TraceTriePointer:
    def __init__(self, node: TrieNode, opidx: int, threshold = None):
        self.node = node
        self.opidx = opidx
        self.tokens = []
        self.threshold = threshold
        self.threshold_idx = None
        self.depth = 0

    def advance(self, token) -> bool:
        # Importantly, we can't check node.is_end here, because if
        # a trie node is a prefix of another string in the trie, we'd
        # break off here when we should keep searching.
        if token not in self.node.children:
            return False
        self.node = self.node.children[token]
        self.tokens.append(token)
        self.depth += 1
        return True

    def complete(self, to_update=None):
        if self.node.is_end and self.opidx >= self.node.opidx:
            self.node.num_visits += 1
            if to_update is not None:
                to_update[tuple(self.tokens)] += 1
            if self.threshold is not None and self.node.num_visits >= self.threshold:
                assert(len(self.tokens) == self.depth)
                return self.tokens, self.opidx
        return None

    # Higher score means better to pick this node.
    def score(self) -> int:
        # Plus 1 to ensure that we don't get owndd by the zero.
        return (self.node.num_visits + 1) * self.depth



# Notes for future Rohan:
# We'll likely want to have a data structure like this to maintain
# all of the traces we have that we are either trying to actively
# record or replay. Since that data structure will have active traverals
# through it while we are potentially adding or removing new traces
# to or from it, we don't want to mess up those existing traversals.
# Something that we can do is have an "age" on each end string, which is
# the operation index at which that trace was first entered into the
# data structure. Then matches only occur if an end node was reached with
# an operation index less than or equal to the operation index that the
# traversal pointer was created with.
#
# There is a question about if I want to make the trace TraceOccurreenceWatcher
# act more like the TraceReplayWatcher, where I don't consider two traces as "seen"
# if the consumption of one means that I don't see the second.
class TraceOccurrenceWatcher:
    def __init__(self, threshold=None):
        self.trie = Trie()
        self.active_pointers = []
        self.threshold = threshold
        self.count = 0
        self.trace_counts = {}

    def reset(self):
        self.trie = Trie()
        self.active_pointers = []
        self.count = 0

    def insert(self, trace, opidx):
        assert(len(trace) >= TRACE_MIN_LENGTH)
        self.trie.insert(trace, opidx)
        self.count += 1
        self.trace_counts[tuple(trace)] = 0

    def remove(self, trace):
        # TODO (rohany): The remove call here isn't checking that the string actually
        #  exists within the data structure, so the count operation is not correct if
        #  that assertion doesn't hold.
        self.trie.remove(trace)
        self.count -= 1

    def process_operation(self, token, opidx):
        self.active_pointers.append(TraceTriePointer(self.trie.root, opidx, self.threshold))
        new_pointers = []
        thresholded = []
        for pointer in self.active_pointers:
            if not pointer.advance(token):
                continue
            new_pointers.append(pointer)
            completed = pointer.complete(to_update=self.trace_counts)
            if completed is not None:
                thresholded.append(completed)
        self.active_pointers = new_pointers
        return thresholded


# TODO (rohany): I'm imagining another kind of TrieWatcher that can be used for us to
#  determine when we should insert a trace record operation into the stream.
#  At a high level, we buffer operations ourselves into a local buffer before issuing
#  them through the Legion pipeline. In front of the Legion pipeline, we have a kind
#  of trace watcher that maintains active pointers into the trie for our buffer. Then,
#  our trace watcher advances trie pointers in order of operation index. Then, as soon as
#  pointer at index `i` has no potential matches, the buffer is flushed up until index i
#  through to the earliest possible index held by trie pointer with index > i. This flushes
#  as much through the pipeline as possible as soon as a trace is not possible to be matched.
#
#  This approach though has a issue with tries that have prefixes of strings inside the trie.
#  I think the cleanest thing is to not allow prefixes, and just have the traces we have
#  recorded / accepted not have prefixes? Because otherwise there's a problem of seeing a
#  prefix so far that matches a trace, should you keep waiting for the rest or not?

# What I'm stuck on right now is the question of a trace being a substring (not necessarily a prefix)
# of another trace in the trie. The prefix problem was one such symptom of it that I ruled out,
# but substrings might also be an issue. I can either complicate the data structure to handle them,
# or define them away. It might be better to just handle them?
# The problem is that it's not just the "deepest" trace that we want to check for completion
# at each new processed operation. Consider the operation stream ABABCDCDABAB, where we are
# watching for the traces ABABCDCDABAB and CDCD. After processing ABABACDCD, we'd have a pointer
# at ABABCDCD, and a pointer at CDCD that is complete. Depending on what we see next, we might
# consider to continue going with ABABCDCDABAB, or just ship the ABAB untraced, ship CDCD traced.
# However, if we have ABAB as a trace as well, then we also might consider just doing ABAB?
#
# I think prefixes and substrings have different complexities. Prefixes likely complicate the
# invariants of the data structure I have in my head.
#
# Spitballing here, I think that the structure should maintain the following things:
# 1) a buffer of pending operations
# 2) a list/set of active pointers into the trie (sorted by depth?)
# 3) a list/set of "completed" pointers in the trie (sorted by depth?)
#
# I think we want to make the heuristic to favor longer traces whenever possible.
# So whenever we see a new token, we advance all active pointers, then see if any
# have completed. If any have completed, add them to the completed set.
# Next, update the op buffer -- flush any operations deeper in the buffer than the
# deepest completed or active pointer.
# Then, compare the set of completed pointers to the set of active pointers.
# If there are any active pointers deeper than the depths of the completed pointers,
# keep going. If there are any completed pointers deeper than active pointers, launch
# the trace corresponding to the deepest one, flush the buffer up to that point, and
# remove all active pointers and completed pointers.
#
# Correction to above. "Depth" is not necessarily the right ordering metric, or not
# by itself. We want "opidx" and "depth". The "opidx" difference between the active
# pointers tells us exactly how many operations we can issue before stepping on
# some other pointer's toes.
#
# I think that we can handle prefixes by maintaining an interval-tree like data structure.
# I don't want to do this yet until we have to, because of this extra cost.

class TraceReplayTrieWatcher:
    def __init__(self):
        # TODO (rohany): I think that this would be like a std::list, rather than a vector.
        self.op_buffer = []
        # The starting index in the operation stream that this buffer corresponds to. It
        # will be advanced once operations are flushed to the "runtime".
        self.op_buf_start_idx = 0
        # These are ordered by depth.
        self.active_watchers = []
        self.completed_watchers = []
        self.trie = Trie()

    def process_operation(self, token, opidx):
        self.op_buffer.append(token)
        # TODO (rohany): Hack setting threshold for the TraceTriePointer so that it
        #  actually returns something if complete succeeds.
        self.active_watchers.append(TraceTriePointer(self.trie.root, opidx, threshold=0))

        advanced_pointers = []
        # TODO (rohany): We're accidentally throwing away completed pointers here?
        #  This should be self.completed_watchers...
        # completed_pointers = []
        completed_pointers = self.completed_watchers
        for pointer in self.active_watchers:
            if pointer.advance(token):
                # Depending on if the pointer is complete, add it to the right list.
                # Different logic will need to be used if we want to support prefixes.
                if pointer.complete() is not None:
                    completed_pointers.append(pointer)
                else:
                    advanced_pointers.append(pointer)


        earliest_active = min([pointer.opidx for pointer in advanced_pointers], default=None)
        earliest_completed = min([pointer.opidx for pointer in completed_pointers], default=None)
        earliest_opidx = min([pointer.opidx for pointer in itertools.chain(advanced_pointers, completed_pointers)], default=None)

        # print("active", len(advanced_pointers), "completed", len(completed_pointers), "earliest active", earliest_active, "earliest completed", earliest_completed, "earliest opidx", earliest_opidx, "op buf start idx", self.op_buf_start_idx, "op buffer", self.op_buffer)

        # No matter what, we can flush all operations before the earliest active or completed pointer.
        # Note that if there are no active pointers or completed pointers, min will be None, flushing
        # the entire buffer. This will change if we want to support prefixes.
        self._flush_buffer(earliest_opidx)

        if earliest_active is None and earliest_completed is None:
            # If we have no active or completed pointers, then there isn't
            # anything left to do.
            pass
        elif earliest_active is None:
            # In this case, we only have completed pointers and no active pointers.
            # So, flush through as many completed operations as we can. This is heuristic
            # to try and issue the largest traces possible. There are potentially other
            # kinds of orderings that could be done here, like a binpack of the depths
            # and starting indices, but I don't think this situation should arise that often anyway?
            sorted_completions = reversed(sorted(completed_pointers, key=lambda pointer: pointer.score()))
            for completed in sorted_completions:
                if completed.opidx < self.op_buf_start_idx:
                    continue
                self._flush_buffer(completed.opidx)
                print(f"1 ISSUING TRACE OF LENGTH {completed.depth} recorded at opidx {completed.node.opidx}, issuing at {opidx}")
                self._flush_buffer(completed.opidx + completed.depth)

            # At this point we've issued all of the possible completed pointers, with no active
            # pointers left in the trie, so issue the rest of the buffer now.
            self._flush_buffer()
            completed_pointers = []
        elif earliest_completed is None:
            # There are no completed pointers, so all there's left to do issue everything
            # that is unmatched now. We can do this by finding the smallest active opidx
            # and flushing the buffer until then. This was also done outside of this case above.
            ...
        else:
            # In the final case, we have completions and active pointers. First, flush
            # through operations farther behind the earliest active or completed pointer,
            # which was done above. Next, if there are completed pointers earlier than
            # active pointers, flush those completed pointers and remove any of the
            # active pointers that are no longer valid.
            if earliest_completed < earliest_active:
                # TODO (rohany): An idea here is that once we have traces recorded and replayed,
                #  we should prefer replaying them versus trying out some new traces (this is an
                #  exploration vs exploitation tradeoff I'm choosing to not make right now). If
                #  we order by visits and opidx, we can avoid randomly taking a long trace that
                #  hasn't been seen much before over a trace we already have recorded and ready.
                #  If this is done, the `completed_pointers` list needs to be updated without
                #  a slice but instead a full filter about everything that is no longer valid.
                #  We also have to make sure that whatever is chosen is before the cutoff point.
                sorted_completions = reversed(sorted(completed_pointers, key=lambda pointer: pointer.score()))
                # sorted_completions = sorted(completed_pointers, key=lambda pointer: pointer.opidx)
                cutoff_opidx = earliest_active
                new_completions = []
                for completed in sorted_completions:
                    if completed.opidx >= cutoff_opidx:
                        new_completions.append(completed)
                        continue
                    if completed.opidx < self.op_buf_start_idx:
                        continue
                    self._flush_buffer(completed.opidx)
                    print(f"2 ISSUING TRACE OF LENGTH {completed.depth} recorded at opidx {completed.node.opidx}, issuing at {opidx}")
                    self._flush_buffer(completed.opidx + completed.depth)
                completed_pointers = new_completions

                # cutoff_opidx = earliest_active
                # for i, completed in enumerate(sorted_completions):
                #     if completed.opidx >= cutoff_opidx:
                #         print("BREAKING HERE?")
                #         completed_pointers = completed_pointers[i:]
                #         break
                #     if completed.opidx < self.op_buf_start_idx:
                #         continue
                #     self._flush_buffer(completed.opidx)
                #     print(f"2 ISSUING TRACE OF LENGTH {completed.depth} recorded at opidx {completed.node.opidx}, issuing at {opidx}")
                #     self._flush_buffer(completed.opidx + completed.depth)
                # Remove any invalid advanced pointers now.
                new_advanced_pointers = []
                for pointer in advanced_pointers:
                    if pointer.opidx >= self.op_buf_start_idx:
                        new_advanced_pointers.append(pointer)
                advanced_pointers = new_advanced_pointers
            elif earliest_completed > earliest_active:
                # If there are active pointers earlier than completed pointers,
                # then there's nothing left to do. In this approach, we are heuristically
                # favoring longer traces over shorter ones.
                ...
            else:
                assert False

        self.active_watchers = advanced_pointers
        self.completed_watchers = completed_pointers

    def insert(self, trace, opidx):
        self.trie.insert(trace, opidx)

    # TODO (rohany): We won't worry about removals for now.

    def _flush_buffer(self, opidx=None):
        if opidx is None:
            self.op_buf_start_idx += len(self.op_buffer)
            self.op_buffer = []
            return
        if self.op_buf_start_idx > opidx:
            return
        real_index = opidx - self.op_buf_start_idx
        self.op_buffer = self.op_buffer[real_index:]
        self.op_buf_start_idx += real_index


class StableTraceReplayTrieWatcher(TraceReplayTrieWatcher):

    # Variant/Copy of the original TraceTriePointer
    class Pointer:
        def __init__(self, node: TrieNode, opidx: int):
            self.node = node
            self.opidx = opidx
            self.depth = 0

        def advance(self, token) -> bool:
            # Importantly, we can't check node.is_end here, because if
            # a trie node is a prefix of another string in the trie, we'd
            # break off here when we should keep searching.
            if token not in self.node.children:
                return False
            self.node = self.node.children[token]
            self.depth += 1
            return True

        def complete(self):
            return self.node.is_end and self.opidx >= self.node.opidx

        def replay(self):
            self.node.num_visits += 1

        # Higher score means better to pick this node.
        def score(self) -> int:
            # Plus 1 to ensure that we don't get owndd by the zero.
            return (self.node.num_visits + 1) * self.depth



    # Mostly copied, but partially overridden method from the parent class.
    def process_operation(self, token, opidx):
        self.op_buffer.append(token)
        self.active_watchers.append(self.Pointer(self.trie.root, opidx))

        advanced_pointers = []
        completed_pointers = self.completed_watchers
        for pointer in self.active_watchers:
            if pointer.advance(token):
                # Depending on if the pointer is complete, add it to the right list.
                # Different logic will need to be used if we want to support prefixes.
                if pointer.complete():
                    completed_pointers.append(pointer)
                else:
                    advanced_pointers.append(pointer)


        earliest_active = min([pointer.opidx for pointer in advanced_pointers], default=None)
        earliest_completed = min([pointer.opidx for pointer in completed_pointers], default=None)
        earliest_opidx = min([pointer.opidx for pointer in itertools.chain(advanced_pointers, completed_pointers)], default=None)

        # No matter what, we can flush all operations before the earliest active or completed pointer.
        # Note that if there are no active pointers or completed pointers, min will be None, flushing
        # the entire buffer. This will change if we want to support prefixes.
        self._flush_buffer(earliest_opidx)
        if earliest_active is None and earliest_completed is None:
            # If we have no active or completed pointers, then there isn't
            # anything left to do.
            pass
        elif earliest_active is None:
            # In this case, we only have completed pointers and no active pointers.
            # So, flush through as many completed operations as we can. This is heuristic
            # to try and issue the largest traces possible. There are potentially other
            # kinds of orderings that could be done here, like a binpack of the depths
            # and starting indices, but I don't think this situation should arise that often anyway?
            sorted_completions = reversed(sorted(completed_pointers, key=lambda pointer: pointer.score()))
            for completed in sorted_completions:
                if completed.opidx < self.op_buf_start_idx:
                    continue
                self._flush_buffer(completed.opidx)
                completed.replay()
                print(f"1 ISSUING TRACE OF LENGTH {completed.depth} recorded at opidx {completed.node.opidx}, issuing at {opidx}")
                self._flush_buffer(completed.opidx + completed.depth)

            # At this point we've issued all of the possible completed pointers, with no active
            # pointers left in the trie, so issue the rest of the buffer now.
            self._flush_buffer()
            completed_pointers = []
        elif earliest_completed is None:
            # There are no completed pointers, so all there's left to do issue everything
            # that is unmatched now. We can do this by finding the smallest active opidx
            # and flushing the buffer until then. This was also done outside of this case above.
            ...
        else:
            # In this case, we have some active pointers and some completed pointers. In this case,
            # we're going to take a different heuristic than the base TraceReplayTrieWatcher where
            # we're only going to flush completed pointers that do not overlap with any active pointers.
            # TODO (rohany): The stable trace replay watcher can only increment the visit count of its
            #  trie nodes when they are flushed, not when they are just completed. We probably need to
            #  just write a different visit class for this instead of trying to co-opt the one again.
            if earliest_completed < earliest_active:
                sorted_completions = reversed(sorted(completed_pointers, key=lambda pointer: (pointer.score(), -pointer.opidx)))
                cutoff_opidx = earliest_active
                new_completions = []
                for completed in sorted_completions:
                    # TODO (rohany): I think that this should be > not >=, but need to test further.
                    if completed.opidx + completed.depth >= cutoff_opidx:
                        new_completions.append(completed)
                        continue
                    if completed.opidx < self.op_buf_start_idx:
                        continue
                    self._flush_buffer(completed.opidx)
                    completed.replay()
                    print(f"2 ISSUING TRACE OF LENGTH {completed.depth} recorded at opidx {completed.node.opidx}, issuing at {opidx}")
                    self._flush_buffer(completed.opidx + completed.depth)
                completed_pointers = new_completions

                # Since we waited to not cutoff any active pointers, there should not be any invalid active pointers.
                # Let's assert this to be safe.
                for pointer in advanced_pointers:
                    assert(pointer.opidx >= self.op_buf_start_idx)
            elif earliest_completed > earliest_active:
                # There are active pointers behind our completed pointers, so there's no point in even looking at anything.
                ...
            else:
                assert(False)

        self.active_watchers = advanced_pointers
        self.completed_watchers = completed_pointers
        # print(len(self.completed_watchers))


def autotrace_state_machine(winnowed, state):
    def p(s, node):
        print(node.num_visits, s[:10], len(s), node.opidx)

    class MachineState(Enum):
        SEARCHING = 0
        # TODO (rohany): There's a way to potentially simplify this
        #  where we get rid of the watching state, and only have the
        #  REPLAYING state, and just treat "waiting for enough visits"
        #  to be the first "tier" of the JIT.
        WATCHING = 1
        REPLAYING = 2

    to_add = 20
    # TODO (rohany): Make the search width a command line parameter.
    # It looks like the winnowing batched processor is strictly better than the
    # winnowing-only processor.
    if winnowed:
        searcher = WinnowingBatchedTraceProcessor(300)
    else:
        searcher = BatchedTraceProcessor(state.prog, 300)

    # Low visit thresholds (like 5) seem to result in traces we don't actually care about
    # getting comitted.
    visit_threshold = 15
    watcher = TraceOccurrenceWatcher(threshold=visit_threshold)

    # MachineState.WATCHING variables.
    watching_count = 0
    watching_longest_added = 0
    watching_last_best = 0
    # End MachineState.WATCHING variables.

    replayer = TraceReplayTrieWatcher()

    # Initial state is "searching".
    jitstate = MachineState.SEARCHING
    for opidx, op in enumerate(state.prog):
        if jitstate == MachineState.SEARCHING:
            traces = searcher.process_operation(op)
            if traces is None:
                # We haven't found any traces yet, so we stay in the searching state.
                jitstate = MachineState.SEARCHING
            else:
                print("GOT SOME TRACES", opidx)
                # We found some traces, so add them to the set of watched traces.
                # TODO (rohany): I'm assuming that we're not trying to have
                #  multiple of these stages happening simulatenously.
                count = 0
                for trace in [[hash(op) for op in trace] for trace in traces]:
                    if len(trace) < TRACE_MIN_LENGTH:
                        continue
                    assert (len(trace) >= TRACE_MIN_LENGTH)
                    # if not watcher.trie.prefix(trace) and not watcher.trie.superstring(trace):
                    if not watcher.trie.prefix(trace):
                        watcher.insert(trace, opidx)
                        watching_longest_added = max(watching_longest_added, len(trace))
                        count += 1
                        if count == to_add:
                            break
                searcher.reset()
                jitstate = MachineState.WATCHING
        elif jitstate == MachineState.WATCHING:
            traces = watcher.process_operation(hash(op), opidx)
            watching_count += 1
            if len(traces) == 0:
                # TODO (rohany): Need to have a fraction here to go back to searching
                #  if we don't upgrade any traces to commit in a *reasonable* amount of time.
                # TODO (rohany): Well how should we do this? If we decide to watch some traces
                #  and none of them get executed a fixed number of times, then we should bail on
                #  this batch. If the maximum length trace is of length N and we wait for (arbitrarily)
                #  5N operations without seeing at least an improvement of (arbitrarily) 3 since last time,
                #  then we've picked the traces badly and should try again.
                # TODO (rohany): I think that we should be in a "mixed" state where we are always watching
                #  potential candidates and replaying traces, and watched traces can be upgraded into replayable
                #  traces when they are caught enough. However, if we decide to do this, then we've basically
                #  just become the merged / tiered stage. Doing this in a single replay data structure is the
                #  problem of counting a single token towards multiple traces or not. Counting towards multiple
                #  traces is nice because we can then get a 100% accurate reading of what traces are being
                #  issued by the application. The problem however is that this is not reality -- two traces can
                #  hit the threshold but be partially overlapping. If we comitted both of them, then we'd never
                #  actually get to replay them both. If we use the "one-token-used-once" behavior of the replayer,
                #  then we would see what traces are actually getting used. The problem is now that a sane implementation
                #  of the tracker needs to encode some kind of heuristics about what traces to choose when given a prefix
                #  so far (i.e. should it wait to see more and potentially match a longer trace or commit to something).
                #  When succumbing to a heuristic like this, we might not get accurate counts of traces seen in the program.
                #  However, with watching being in the same state as replaying, what could happen, and what was
                #  happening before, is that we could be happily replaying trace A, and then decide that hey, it's time
                #  to commit a new trace B which ends in a prefix of A. Then we'd all of a sudden stop replaying A and
                #  switch over to B which is not going to be the right choice. This won't happen in steady state if we
                #  treat watched traces in the replayable data structure, because the set of watched traces will
                #  hit a steady state with one another, but a newly inserted trace could lead to a problem.
                # TODO (rohany): I think that I want to just go ahead and make an implementation of the trace trie
                #  replay watcher that can handle the waiting? This is all very hazy, but I want to discuss it with Mike.
                # SOOO -- Questions / discussion points for mike:
                #  1) state machine versus always-on approach
                #  2) how to handle tiering? one-at-atime or precise approach?
                #  3) trace selection issue -- complexity of it tradeoff versus some unpredictable behavior
                #     * could lead to bad behavior with chains of traces picked, AB; BC; CD; DE ... Could wait a long time!
                #  4) Some of this complexity goes away if we just consider one  active trace at a time. Is this too restrictive?
                jitstate = MachineState.WATCHING
                if watching_count >= 5 * watching_longest_added:
                    best = 0
                    for _, v in watcher.trace_counts.items():
                        best = max(best, v)
                    if best - watching_last_best < 3:
                        print("GOING BACK TO SEARCHING")
                        # Jump back to searching.
                        jitstate = MachineState.SEARCHING
                        watching_count = 0
                        watching_last_best = 0
                        watching_longest_added = 0
                    else:
                        # Stay in the watching state.
                        jitstate = MachineState.WATCHING
                        watching_count = 0
            else:
                # TODO (rohany): This is right now structured to escalate a single trace
                #  trace from the watched to committed tries. It's not built right now to
                #  do multiple.
                print("UPDATING TRACES TO COMMITTING", opidx)
                for trace, idx in traces:
                    if not (replayer.trie.prefix(trace) or replayer.trie.superstring(trace)):
                        replayer.insert(trace, idx)
                watcher.reset()
                jitstate = MachineState.REPLAYING
        elif jitstate == MachineState.REPLAYING:
            # TODO (rohany): Need to have a heuristic drop us out of the
            #  replaying state if we aren't replaying enough operations.
            replayer.process_operation(hash(op), opidx)
            jitstate = MachineState.REPLAYING
        else:
            assert False


    # assert (committed.op_buf_start_idx + len(committed.op_buffer) == len(state.prog))


def autotrace_always_on(winnowed, state, batchsize):
    def p(s, node):
        print(node.num_visits, s[:10], len(s), node.opidx)

    to_add = 10
    # to_remove = 5
    # Low visit thresholds (like 5) seem to result in traces we don't actually care about
    # getting comitted.
    visit_threshold = 15
    maximum_watching = 25

    # It looks like the winnowing batched processor is strictly better than the
    # winnowing-only processor.
    if winnowed:
        processor = WinnowingBatchedTraceProcessor(batchsize)
    else:
        processor = BatchedTraceProcessor(state.prog, batchsize)

    watcher = TraceOccurrenceWatcher(threshold=visit_threshold)
    # committed = TraceReplayTrieWatcher()
    committed = StableTraceReplayTrieWatcher()
    for opidx, op in enumerate(state.prog):
        for trace, idx in watcher.process_operation(hash(op), opidx):
            assert(len(trace) >= TRACE_MIN_LENGTH)
            # TODO (rohany): I haven't yet thought of the cleanest way to check this, but I also
            #  don't want to insert something if it is a superstring of a trace that
            #  is already in the tree, because then we have a prefix situation?
            # if not (committed.trie.prefix(trace) or committed.trie.superstring(trace)):
            if not committed.trie.prefix(trace):
                committed.insert(trace, idx)
        committed.process_operation(hash(op), opidx)

        traces = processor.process_operation(op)
        if traces is not None:
            if watcher.count + to_add > maximum_watching:
                # Remove potential traces with the fewest number of visits.
                all_traces = []
                def add_trace(s, node):
                    all_traces.append((s, node.num_visits))
                watcher.trie.foreach_string(add_trace)
                for trace, _ in sorted(all_traces, key=lambda x: x[1])[:(watcher.count + to_add - maximum_watching)]:
                    watcher.remove(trace)

            # TODO (rohany): Insertions should go before the deletions ... Though I guess I did
            #  this originally so that we wouldn't just delete the things we add immediately because
            #  they have 0 count?
            count = 0
            for trace in [[hash(op) for op in trace] for trace in traces]:
                if len(trace) < TRACE_MIN_LENGTH:
                    continue
                assert (len(trace) >= TRACE_MIN_LENGTH)
                # if not watcher.trie.prefix(trace) and not watcher.trie.superstring(trace):
                if not watcher.trie.prefix(trace):
                    watcher.insert(trace, opidx)
                    count += 1
                    if count == to_add:
                        break

    print("Final committed traces:")
    committed.trie.foreach_string(p)

    assert (committed.op_buf_start_idx + len(committed.op_buffer) == len(state.prog))


def main(filename, winnowed, state_machine, batchsize):

    # winnower = Winnower(10, 4)
    # for tok in "adorunrunrunadorunrun":
    #     winnower.process_operation(tok)
    # exit(0)

    # watcher = TraceReplayTrieWatcher()
    # traces = ["ABABCDCD", "CDCD", "BABACDCDA"]
    # for idx, s in enumerate(traces):
    #     watcher.insert(s, idx)
    #
    # for idx, tok in enumerate("CDCDABABABABCDCDABABCDCD"):
    #     watcher.process_operation(tok, idx + len(traces))
    #
    # exit(0)

    # watcher = Trie()
    # for s in ["ABCD", "ABY", "AZ"]:
    #     watcher.insert(s, 0)
    # print(watcher.superstring("AZ"))
    # exit(0)

    # def p(s, node):
    #     print(s, node.num_visits)
    # watcher = Trie()
    # for s in ["ABCD", "DEF", "ABC"]:
    #     watcher.insert(s, 0)
    #
    # watcher.remove("DEF")
    # watcher.remove("ABCD")
    # watcher.remove("ABC")
    #
    # watcher.foreach_string(p)
    #
    # # for c in "ABCABCDEFABCDABCDABCDEFDEF":
    # #     watcher.process_operation(c)
    # # def p(s, node):
    # #     print(s, node.num_visits)
    # # watcher.trie.foreach_string(p)
    # exit(0)

    # TODO (rohany): What are the actual interfaces that I want to expose here?
    #  1) TraceProcessor -- a component that processes a stream an operation at a time,
    #     and maybe returns a list of traces to start considering.
    #  2) TraceWatcher -- a component that ingests traces from the TraceProcessor and watches
    #     the operation stream to decide when to start recording a trace.
    #  3) ActiveTraceManager -- a component that maintains traces committed to by the
    #     TraceWatcher and tracks how many times they have been hit by the application,
    #     essentially seeing the results of the decisions we have made.

    print("Loading file...")
    with open(filename, 'r') as f:
        state = parse_spy_log(f)
    state.prune_ops()
    print(f"Loaded Legion Spy log, {len(state.prog)} ops")

    if state_machine:
        autotrace_state_machine(winnowed, state)
    else:
        autotrace_always_on(winnowed, state, batchsize)


    # print("Final watcher traces:")
    # watcher.trie.foreach_string(p)

    # t = time.time()
    # candidates = find_repeat_candidates(state.prog, RepeatAlgorithms.LONGEST_NONOVERLAPPING_REPEAT)
    # print("compute time: ", time.time() - t)
    #
    # for candidate in candidates:
    #     print(candidate.end - candidate.start, candidate.num_repeats)
    #
    # best = candidates[0]
    #
    # op1 = tuple(S[best.start:best.end])
    #
    # print(f"BEST RepeatLen={len(op1)}")
    # count = 0
    # lastmatch = None
    # for i in range(best.end, len(state.prog)):
    #     if op1 == tuple(S[i:i+(best.end-best.start)]):
    #         print("Matched at index: ", i)
    #         count += 1
    #         if lastmatch is not None:
    #             assert(i - lastmatch >= len(op1))
    #         lastmatch = i
    # print(f"Total matched: {count}")

    #
    # op2 = state.prog[best.end:best.end + (best.end-best.start)]
    # assert(tuple(op1) == tuple(op2))

    #
    # def walk(node):
    #     if isinstance(node, Internal):
    #         print(node.end - node.start, len(node.children), node.start, node.end, node)
    # tree.pre_order(walk)
    # print(tree.root.to_dot())

    # _, results = another_tandem_repeat(S)
    # for start, width, repeats in results:
    #     print(f"RepeatLen: {width}, NumRepeats: {repeats}")# , S[start:start+width])

    # print("Naive tandem repeats")
    # _, results = naivetrepeats(S)
    # for start, width, repeats in results:
    #     print(f"RepeatLen: {width}, NumRepeats: {repeats}")# , S[start:start+width])
    #
    # # _, results = tandem5(S)
    # # for start, width, repeats in results:
    # #     print(f"RepeatLen: {width}, NumRepeats: {repeats}")# , S[start:start+width])
    #
    # sarray = suffix_array(S)
    # lcp = lcp_array(S, sarray)
    # repeats = tandem_repeats(S, sarray, lcp)
    # for r, (s, n) in repeats.items():
    #     print(f"num_repeats={n}, repeat_len={len(r)}")
    # num, start, repeat = list(reversed(sorted([(num, start, repeat) for repeat, (start, num) in repeats.items()])))[0]
    # prog_frag = state.prog[start : start+len(repeat)]
    # print(f"Identified trace repeats={num}, len={len(repeat)}")
    # for op in prog_frag:
    #     print(op)


parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("--winnowed", default=False, action="store_true")
parser.add_argument("--state_machine", default=False, action="store_true")
parser.add_argument("--batchsize", default=300, type=int)
args = parser.parse_args()

if __name__ == "__main__":
    main(**vars(args))