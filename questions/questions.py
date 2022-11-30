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

def binary_search(lst, value, left=None, right=None):
    """
    Return index of value in sorted list.
    If value is not present or list is empty, return -1

    Keyword arguments:
    lst -- a sorted list
    value -- value to locate in list
    left (optional, default None) -- left bound of index range
    right (optional, default None) -- right bound of index range
    """

    # Set initial index range, handle empty list
    if left is None and right is None:
        right = len(lst)

        # Return -1 for empty list
        if right == 0:
            return -1

        left = 0

    # Get middle index of the index range
    mid = (left + right) // 2

    # Base case: last item in index range is not the value
    if right - left <= 1 and lst[mid] != value:
        return -1

    # Base case: found value at mid index, return index
    if lst[mid] == value:
        return mid

    # Recursive case: value checked was greater than arg value
    elif lst[mid] > value:
        return binary_search(lst, value, left=left, right=(mid-1))

    # Recursive case: value checked was less than arg value
    elif lst[mid] < value:
        return binary_search(lst, value, left=(mid+1), right=right)

class MyNode:

    def __init__(self, value=None, link=None):
        self.value = value
        self.link = link

    def __str__(self):
        return f"Node with value:{self.value}"


class MySLL:

    def __init__(self,
                 head=None,
                 tail=None):

        self.head = head
        self.tail = tail
        self._length = 0

    def __str__(self):
        return f"Sll with length {self.length}"

    # Length instance attribute
    # "read-only" length property
    @property
    def length(self):
        return self._length

    # Traverse
    def traverse(self, idx):

        # handle negative idx
        if idx < 0:
            idx == self._length + idx

        # If list empty
        # If idx out of bounds
        if (
            self._length == 0 or
            idx > (self._length - 1)
        ):
            raise IndexError("idx out of range")

        # If idx not an integer
        if not isinstance(idx, int):
            raise TypeError("idx must be an integer")

        i = 0
        _current_node = self.head

        while i < idx:
            _current_node = _current_node.link
            i += 1

        return _current_node

    def insert(self, val, idx=None):

        _new_node = MyNode(val, link=None)

        # idx None (default)
        if idx is None:
            idx = self._length

        # handle negative indices
        if (idx < 0):
            idx = self._length + idx

        # list empty, inserting first node
        if self.head is None:
            self.head = _new_node
            self.tail = _new_node
            self._length += 1
            return None

        # insert at tail (default)
        elif (idx == self._length):
            self.tail.link = _new_node
            self.tail = _new_node
            self._length += 1
            return None

        # insert before head
        elif idx == 0:
            _new_node.link = self.head
            self.head = _new_node
            self._length += 1
            return None

        # insert between two nodes
        else:

            this_idx_node = self.traverse(idx - 1)
            next_idx_node = this_idx_node.link

            _new_node.link = next_idx_node
            this_idx_node.link = _new_node

            self._length += 1
            return None

    def remove(self, idx=0):

        # If list empty
        # If idx out of bounds
        if (
            self._length == 0 or
            idx > (self._length - 1)
        ):
            raise IndexError("idx out of range")

        # If idx not an integer
        if not isinstance(idx, int):
            raise TypeError("idx must be an integer")

        # If idx 0
        if idx == 0:
            _val = self.head.value
            self.head = self.head.link
            self._length -= 1

            # If no other members
            if self._length == 0:
                self.tail = None

            return _val
        while idx < 0:
            idx += self._length

        # other indices
        previous_idx_node = self.traverse(idx - 1)
        this_idx_node = previous_idx_node.link

        _val = this_idx_node.value

        next_idx_node = this_idx_node.link

        # connect previous to next
        previous_idx_node.link = next_idx_node

        if next_idx_node is None:
            self.tail = previous_idx_node

        self._length -= 1
        return _val

    def search(self, given_val):
        """
        search
        if given_val is in the sll, return the idx
        of the first occurrence
        else return -1
        """
        _current_node = self.head
        for i in range(self._length):
            if _current_node.value == given_val:
                return i
            _current_node = _current_node.link
        return -1


def find_lists_with_minimum_value(lists):
    mins = [min(lst) for lst in lists]
    min_of_mins = min(mins)
    return [i for i, n in enumerate(mins) if n == min_of_mins]

def find_lists_with_minimum_value(lists):
    result = []
    minimum = float('inf')
    for nums in lists:
        for num in nums:
            if num < minimum:
                minimum = num
    for i, nums in enumerate(lists):
        if minimum in nums:
            result.append(i)
    return result

def find_lists_with_minimum_value(lists):
    minimum = min([
        min(nums)
        for nums in lists
    ])
    return [
        i
        for i, nums in enumerate(lists)
        if minimum in nums
    ]

def baby_talk(s):
    max_length = 0
    for i, letter in enumerate(s):
        for j in range(1, len(s) // 2 + 1):
            if s[i:j+i] == s[j+i:j+(j+i)]:
                max_length = max(max_length, j*2)
    return max_length

def baby_talk(s):
    window = len(s)
    if window % 2 == 1: window -= 1
    while window > 0:
        for i in range(0, len(s) - window + 1):
            substr = s[i:i+window]
            half = window // 2
            first = substr[:half]
            second = substr[half:]
            if first == second:
                return window
        window = window - 2
    return window


def closest_to_zero(nums):
    m=100000000
    for i in nums:
        if abs(m)>=abs(i):
            if abs(m)==abs(i):
                if i>m:
                    m=i
            else:
                m=i
    return m

def closest_to_zero(nums):
    best_value = nums[0]
    for value in nums[1:]:
        if abs(value) < abs(best_value):
            best_value = value
        elif abs(value) == abs(best_value) and value > best_value:
            best_value = value
    return best_value

# This is what the class looks like that defines the binary
# tree nodes.
# class TreeNode:
#     def __init__(self, value, left=None, right=None):
#         self.value = value
#         self.left = left
#         self.right = right

# root is an instance of the TreeNode class
def check_tree(root):
    sum = root.left.value + root.right.value
    return root.value == sum


def check_tree(root):
    sum_of_child_values = root.left.value + root.right.value
    return root.value == sum_of_child_values

def check_tree(root):
    return root.value == root.left.value + root.right.value

def radix_sort(values):
    # Find the maximum value in the input array
    max_item = max(values)

    # Find the number of digits in the largest value
    digit_count = 1
    while max_item > 0:
        max_item /= 10
        digit_count += 1

    # Start with the rightmost digit for sorting
    tens_place = 1

    # Step 4
    while digit_count > 0:
        # Ten buckets because 10 possible digits
        buckets = [0] * 10
        n = len(values)

        for i in range(n):
            # Go through each value in values and put
            # note that it's in the proper bucket
            sort_digit = (values[i] // tens_place) % 10
            buckets[sort_digit] += 1

        # Do some fancy math so that we know which
        # numbers are in which bucket
        for i in range(1, 10):
            buckets[i] += buckets[i-1]

        # Move the values from the buckets into
        # another list
        sorted_values = [0] * n
        i = n - 1
        while i >= 0:
            item = values[i]
            sort_digit = (values[i] // tens_place) % 10
            buckets[sort_digit] -= 1
            sorted_position = buckets[sort_digit]
            sorted_values[sorted_position] = item
            i -= 1

        # Put the values from the sorted list
        # back into the origin values list
        for i, item in enumerate(sorted_values):
            values[i] = item

        tens_place *= 10
        digit_count -= 1


def partition(values, left, right):
    pivot = values[right]
    star = left - 1
    for i in range(left, right):
        if values[i] <= pivot:
            star += 1
            values[star], values[i] = values[i], values[star]
    star += 1
    values[star], values[right] = values[right], values[star]
    return star

def quicksort(values, left=None, right=None):
    if left is None and right is None:
        left = 0
        right = len(values) - 1
    if left >= right or left < 0:
        return
    p = partition(values, left, right)
    quicksort(values, left, p - 1)
    quicksort(values, p + 1, right)

def merge_sort(values, left=None, right=None):
    if left is None and right is None:
        left = 0
        right = len(values) - 1

    # Base case
    if left >= right:
        return

    # Recursive cases
    # Find the middle to split
    middle = (right + left) // 2

    # Sort the left half
    merge_sort(values, left, middle)

    # Sort the right half
    merge_sort(values, middle + 1, right)

    # Merge them together
    merge(values, left, middle, right)

def merge(values, left, middle, right):
    right_start = middle + 1

    # Terminal case to make sure we don't loop
    # forever
    if values[middle] <= values[right_start]:
        return

    # Merge the sub-lists by looping and comparing
    # the values at the start of each list
    while left <= middle and right_start <= right:
        # The one on the left is less than the
        # one on the right, so just keep going
        if values[left] <= values[right_start]:
            left += 1
        else:
            # In this case, the one on the right half
            # is less than one in the left half, so
            # we need to swap the values
            value = values[right_start]
            index = right_start

            # Move the all of the values to the right
            # by one
            while index != left:
                values[index] = values[index - 1]
                index -= 1

            # Put the value into the new "empty" place
            values[left] = value

            # Increment all of the indexes
            left += 1
            middle += 1
            right_start += 1

def selection_sort(values):
    for i in range(len(values)):
        min_value_index = i
        for j in range(i + 1, len(values)):
            if values[min_value_index] > values[j]:
                min_value_index = j
        values[i], values[min_value_index] = values[min_value_index], values[i]


def bubble_sort(values):
    n = len(values)
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if values[j] > values[j + 1]:
                values[j], values[j + 1] = values[j + 1], values[j]

def linear_search(values, target):
    for i, value in enumerate(values):
        if target == value:
            return i
    return -1

def binary_search(values, target):
    left = 0
    right = len(values) - 1

    while left <= right:
        middle = (left + right) // 2
        if values[middle] < target:
            left = middle + 1
        elif values[middle] > target:
            right = middle - 1
        else:
            return middle

    return -1

def binary_search(values, target, left=None, right=None):
    if left is None and right is None:
        left = 0
        right = len(values) - 1

    # Base case: did not find item
    if left > right:
        return -1

    # Recursive case
    middle = (left + right) // 2
    if values[middle] < target:
        return binary_search(values, target, middle + 1, right)
    elif values[middle] > target:
        return binary_search(values, target, left, middle - 1)
    else:
      return middle

def reverse_prefix(s, letter):
    # Find the index of the letter
    index = s.find(letter)

    # If the letter is not in the string, return
    # the original string
    if index == -1:
        return s

    # Reverse the string up to the letter
    return s[:index + 1][::-1] + s[index + 1:]

def reverse_prefix(s, letter):
    index = -1
    for i, c in enumerate(s):
        if c == letter:
            index = i
            break
    if index == -1:
        return s
    prefix = s[:index + 1]
    prefix_reversed = ''.join(reversed(prefix))
    return prefix_reversed + s[index + 1:]

def reverse_prefix(s, letter):
    index = s.find(letter)
    if index == -1:
        return s
    prefix = s[:index + 1]
    prefix_reversed = ''.join(reversed(prefix))
    return prefix_reversed + s[index + 1:]

def is_double_reversible(num):
    return num == 0 or num % 10 != 0

def is_double_reversible(num):
    s = str(num)
    s = ''.join(reversed(s))
    reversed_once = int(s)

    s = str(reversed_once)
    s = ''.join(reversed(s))
    reversed_twice = int(s)

    return num == reversed_twice


def parse_calculation(s):
    op = s.split(" ")
    return eval(f'{op[1]} {op[0]} {op[2]}')


def find_expression_bounds(starting_index, e) -> int:
    stack_depth = 1
    index = starting_index
    if "(" in e[starting_index]:
        stack_depth += 1
        index += 1
        while stack_depth > starting_index:
            if ")" in e[index]:
                stack_depth -= 1
            elif "(" in e[index]:
                stack_depth += 1
            index += 1
    else:
        index += 1
    return index


def parse_calculation_ext(s):
    e = s.split(" ")
    if "(" not in e[0]:
        return int(e[0])
    else:
        left_bound = find_expression_bounds(1, e)
        left = ' '.join(e[1:left_bound])
        right = ' '.join(e[left_bound:])[:-1]

        if "+" in e[0]:
            return parse_calculation_ext(left) + parse_calculation_ext(right)
        elif "-" in e[0]:
            return parse_calculation_ext(left) - parse_calculation_ext(right)
        elif "*" in e[0]:
            return parse_calculation_ext(left) * parse_calculation_ext(right)
        elif "/" in e[0]:
            return parse_calculation_ext(left) / parse_calculation_ext(right)


def parse_calculation_ext(s):
    start = s.rfind("(")
    stop = start + s[start:].find(")") + 1

    if start == -1:
        return eval(s)

    substring = s[start:stop]
    evaluated_string = s.replace(substring, str(parse_calculation(substring)))
    return parse_calculation_ext(evaluated_string)


def parse_calculation(s):
    v = s.strip("()").split()
    return eval(v[1] + v[0] + v[2])

import re

def calc_match(match):
  s = match.group()
  parts = s.split(' ')
  return str(eval(f'{parts[1]} {parts[0][1]} {parts[2]}'))

s = "(- (+ (+ 2 4 ) (* 1 8 ) ) 15 )"

while '(' in s:
  s = re.sub('\([+-/*] \d+ \d+ \)', calc_match, s)


import re

def calc_match(match):
  s = match.group()
  parts = s.split(' ')
  return str(eval(f'{parts[1]} {parts[0][1]} {parts[2]}'))

s = "(- (+ (+ 2 4 ) (* 1 8 ) ) 15 )"

while '(' in s:
  s = re.sub('\([+-/*] \d+ \d+ \)', calc_match, s)

def count_letters(string):
    count = 0
    for c in string:
        if c == "H":
            count += 1
    return count


def count_letters(string):
    count = 0
    letter_to_look_for = "H"
    for c in string:
        if c == letter_to_look_for:
            if letter_to_look_for == "H":
                letter_to_look_for = "A"
            elif letter_to_look_for == "A":
                letter_to_look_for = "H"
                count += 1
    return count

def count_letters(string):
    letter_sequence = "HACK"
    sequence_index = 0
    count = 0
    for c in string:
        if c == letter_sequence[sequence_index]:
            sequence_index += 1
            if sequence_index == len(letter_sequence):
                count += 1
                sequence_index = 0
    return count

def does_product_exist(nums, left, right, target):
    while left < right:
        product = nums[left] * nums[right]
        if product == target:
            return True
        elif product < target:
            left += 1
        else:
            right -= 1
    return False
import math
def find_larger_area(s, r):
    if s * s > math.pi * r * r:
        return "SQUARE"
    else:
        return "CIRCLE"

def make_integer(nums, length):
    result = []
    for i in range(length):
        result.append(nums[i])
    return result

def does_product_exist(nums, left, right, target):
    for i in range(left, right + 1):
        for j in range(i + 1, right + 1):
            if nums[i] * nums[j] == target:
                return True
    return False

def make_integer(nums, length):
    result = sorted(nums)
    resu = result[0:length]
    s = [str(i) for i in resu]
    res = int("".join(s))
    return res
from functools import reduce


def make_integer(nums, length):
    return reduce(lambda acc, num: acc * 10 + num, sorted(nums)[:length], 0)

def make_integer(nums, length):
    digits = sorted(nums)[:length]
    return int(''.join([str(d) for d in digits]))

def make_integer(nums, length):
    nums = sorted(nums)
    value = 0
    for i in range(length):
        value = value * 10 + nums[i]
    return value

def does_product_exist(nums, left, right, target):
    for i in range(left, right + 1):
        for j in range(i + 1, right + 1):
            if nums[i] * nums[j] == target:
                return True
    return False

def find_larger_area(s, r):
    square_area = s * s
    circle_area = math.pi * r * r
    if square_area > circle_area:
        return "SQUARE"
    else:
        return "CIRCLE"

def make_integer(nums, length):
    result = sorted(nums)
    resu = result[0:length]
    s = [str(i) for i in resu]
    res = int("".join(s))
    return res


def make_integer(nums, length):
    nums.sort()
    small = ''
    for n in nums[0:length]:
        small += str(n)
    return int(small)

def does_product_exist(nums, left, right, target):
    for i in nums[left: right + 1]:
        for j in nums[left: right +1]:
            if i * j == target:
                return True
    return False




def find_odd(nums):
    for i in nums:
        if nums.count(i) % 2 != 0:
            return i

def fix_misspellings(corrections):
    return [
        word[:position-1] + word[position:]
        for word, position
        in [get("word", "position")(correction)
            for correction
            in corrections
        ]
    ]
def fix_misspellings(corrections):
    fixed = []
    for correction in corrections:
        word = correction["word"]
        p = correction["position"]
        fixed_word = word[:p-1] + word[p:]
        fixed.append(fixed_word)
    return fixed

def digits_to_number(digits):
    s = ''.join([str(d) for d in digits])
    return int(s)


def add_digits(digits1, digits2):
    num1 = digits_to_number(digits1)
    num2 = digits_to_number(digits2)
    return num1 + num2


def add_reversed_digits(reversed_digits1, reversed_digits2):
    digits1 = reversed(reversed_digits1)
    digits2 = reversed(reversed_digits2)
    return add_digits(digits1, digits2)


def sum_reversed_digits_as_list(reversed_digits1, reversed_digits2):
    sum = add_reversed_digits(reversed_digits1, reversed_digits2)
    return [int(d) for d in str(sum)]


def sum_reversed_digits_as_reversed_list(reversed_digits1, reversed_digits2):
    l = sum_reversed_digits_as_list(reversed_digits1, reversed_digits2)
    return list(reversed(l))

def sum_reversed_digits_as_reversed_list(reversed_digits1, reversed_digits2):
    return list(reversed(str(int(''.join(str(d) for d in reversed_digits1)) + int(''.join(str(d) for d in reversed_digits2)))))

def pipe_outputs(num_pipes, steps):
    f = num_pipes * [8]
    for s in steps:
        idx = s[0]-1
        if len(s) == 2:
            f.insert(idx+1,(f[idx]-s[1]))
            f[idx] = s[1]
        if len(s) == 1:
            f[idx] = f[idx]+f.pop(idx+1)
    return f
