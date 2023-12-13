#!/usr/bin/env python3

import argparse
import copy
import ctypes
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
                node.num_repeats = count
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


def main(filename):
    print("Loading file...")
    with open(filename, 'r') as f:
        state = parse_spy_log(f)
    state.prune_ops()
    print(f"Loaded Legion Spy log, {len(state.prog)} ops")

    t = time.time()
    best = find_repeat_candidates(state.prog, RepeatAlgorithms.LONGEST_NONOVERLAPPING_REPEAT)[0]
    print("compute time: ", time.time() - t)

    S = [hash(op) for op in state.prog]
    op1 = tuple(S[best.start:best.end])

    print(f"RepeatLen={len(op1)}")
    count = 0
    lastmatch = None
    for i in range(best.end, len(state.prog)):
        if op1 == tuple(S[i:i+(best.end-best.start)]):
            print("Matched at index: ", i)
            count += 1
            if lastmatch is not None:
                assert(i - lastmatch >= len(op1))
            lastmatch = i
    print(f"Total matched: {count}")

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
args = parser.parse_args()

if __name__ == "__main__":
    main(**vars(args))