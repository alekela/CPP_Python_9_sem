balanced_tree = [(1, 13), (2, 8), (3, 17), (4, 1), (5, 11), (6, 15), (7, 25), (9, 6), (14, 22), (15, 27)]
not_balanced_tree = [(1, 10), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (16, 1), (13, 1)]


def check_if_balanced(tree):
    indexes = list(map(lambda x: x[0], tree))
    max_len = len(max(indexes, key=lambda x: len(x)))
    flag = True
    for i in range(1, max_len):
        tmp_indexes = list(filter(lambda x: x, map(lambda x: x[i:], indexes)))
        zero_started = list(filter(lambda x: x[0] == '0', tmp_indexes))
        one_started = list(filter(lambda x: x[0] == '1', tmp_indexes))
        max_len_0 = len(max(zero_started, key=lambda x: len(x)))
        max_len_1 = len(max(one_started, key=lambda x: len(x)))
        if abs(max_len_0 - max_len_1) > 1:
            return False
    return True


def check_order(tree):
    if len(tree) == 1 or len(tree) == 0:
        return True
    center = [item for item in tree if item[0] == '0' or item[0] == '1']
    if len(center):
        center = center[0]
    else:
        center = None
    tree.remove(center)
    tree = [(elem[0][1:], elem[1]) for elem in tree]
    zero_started = list(filter(lambda x: x[0][0] == '0', tree))
    one_started = list(filter(lambda x: x[0][0] == '1', tree))
    left = [item for item in tree if item[0] == '0']
    if len(left):
        left = left[0]
    else:
        left = None
    right = [item for item in tree if item[0] == '1']
    if len(right):
        right = right[0]
    else:
        right = None

    if left and left[1] > center[1]:
        return False
    if right and right[1] < center[1]:
        return False
    return check_order(zero_started) and check_order(one_started)


def prepare_tree(tree):
    indexes = list(map(lambda x: str(bin(x[0]))[2:], tree))
    new_tree = [(indexes[i], tree[i][1]) for i in range(len(indexes))]
    return new_tree


prepared_balanced_tree = prepare_tree(balanced_tree)
answer = check_if_balanced(prepared_balanced_tree)
if answer:
    print("Balanced")
else:
    print("Not balanced")

answer2 = check_order(prepared_balanced_tree)
if answer2:
    print("Ordered")
else:
    print("Not ordered")