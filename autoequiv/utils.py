def dict_to_arr(x):
    import numpy as np
    if isinstance(next(iter(x)), tuple):
        n = max([i for (i, j) in x.keys()]) + 1
        m = max([j for (i, j) in x.keys()]) + 1
        y = np.zeros((n, m), dtype=int)
        for i in range(n):
            for j in range(m):
                y[i, j] = x.get((i, j), -1)
    else:
        n = max([i for i in x.keys()]) + 1
        y = np.zeros((n, 1), dtype=int)
        for i in range(n):
            y[i] = x.get(i, -1)
    return y


def is_permutation(x, n=None):
    ''' is `x` a permutation of {0, 1, ..., `n` - 1}?'''
    if n == None:
        n = len(x)
    elif len(x) != n:
        return False
    s = set(x)
    if len(s) != n:
        return False
    for i in range(n):
        if i not in s:
            return False
    return True


def generate_even_permutations(n):
    if n == 1:
        return [[0]]
    ans = [x + [n - 1] for x in generate_even_permutations(n - 1)]
    odd_perms = generate_odd_permutations(n - 1)
    for perm in odd_perms:
        for i in range(n - 1):
            new_perm = perm + [perm[i]]
            new_perm[i] = n - 1
            ans.append(new_perm)
    return ans


def generate_odd_permutations(n):
    if n == 1:
        return []
    ans = [x + [n - 1] for x in generate_odd_permutations(n - 1)]
    even_perms = generate_even_permutations(n - 1)
    for perm in even_perms:
        for i in range(n - 1):
            new_perm = perm + [perm[i]]
            new_perm[i] = n - 1
            ans.append(new_perm)
    return ans


def generate_all_permutations(n):
    return generate_even_permutations(n) + generate_odd_permutations(n)


def combine_permutations(first_perm, second_perm):
    assert is_permutation(first_perm)
    assert is_permutation(second_perm)
    assert len(first_perm) == len(second_perm)
    n = len(first_perm)
    perm = [0] * n
    for i in range(n):
        perm[i] = second_perm[first_perm[i]]
    assert is_permutation(perm, n)
    return perm


def permutation_of_permutations(permutations, perm):
    n = len(permutations)
    assert perm in permutations
    perm_to_idx = {tuple(perm): i for i, perm in enumerate(permutations)}
    ans = [-1] * n
    for j in range(n):
        perm_src = permutations[j]
        perm_dest = combine_permutations(first_perm=perm, second_perm=perm_src)
        ans[j] = perm_to_idx[tuple(perm_dest)]
    assert is_permutation(ans, n)
    return ans
