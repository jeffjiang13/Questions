def binary_search(items, target):
    start = 0
    end = len(items) - 1
    while start <= end:
        mid = (start + end) // 2
        v = items[mid]
        if target == v:
            return v
        elif target < v:
            end = mid - 1
        else:
            start = mid + 1
    return None

def binary_search(items, target):
    start = 0
    end = len(items) - 1
    while start <= end:
        half = (start + end) // 2
        value = items[half]
        if target == value:
            return value
        elif target < value:
            end = half - 1
        else:
            start = half + 1
    return None
