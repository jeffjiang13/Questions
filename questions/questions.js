function binarySearch(numbers, target) {
    let start = 0;
    let end = numbers.length - 1;

    while (start <= end) {
      const mid = Math.floor((start + end) / 2);
      const guess = numbers[mid];
      if (numbers[mid] === target) {
        return guess;;
      }
      else if (target < guess) {
        end = mid - 1;
      }else {
        start = mid + 1;
      }
    }
    return null;
  }


  function binarySearch(items, target) {
    let start = 0;
    let end = items.length-1;

    while(start <= end) {
        let half = start + Math.floor((end - start) / 2);
        const value = items[half];
        if(target === value) {
            return value;
        } else if(target < value) {
            end = half - 1;
        } else {
            start = half + 1;
        }
    }
    return null;
}

function binary_search(lst, value, left=undefined, right=undefined) {
  /*
  Return index of value in sorted list.
  If value is not present or list is empty, return -1

  Keyword arguments:
  lst -- a sorted list
  value -- value to locate in list
  left (optional, default None) -- left bound of index range
  right (optional, default None) -- right bound of index range
  */

  // Set initial index range, return -1 if lst is empty
  if (left === undefined && right === undefined) {
    right = lst.length;

    // Return -1 for empty lst
    if (right === 0) {
      return -1;
    }

    left = 0;
  }

  // Get middle index of the index range
  var mid = Math.floor((left + right) / 2);

  // Base case: last item in index range is not the value
  if (right - left < 1 && lst[mid] != value) {
    return -1;
  }

  // Base case: found value at mid index, return index
  if (lst[mid] == value) {
    return mid;

  } else if (lst[mid] > value) {

    // Recursive case: value checked was greater than arg value
    return binary_search(lst, value, left=left, right=(mid-1));

  } else if (lst[mid] < value) {

    // Recursive case: value checked was less than arg value
    return binary_search(lst, value, left=(mid+1), right=right);

  }
}



var twoSum = function(nums, target) {

  for (let i = 0; i < nums.length; i++){
      for (let j = i + 1; j < nums.length; j++){
          if(nums[i] + nums[j] === target){
              return [i,j]
          }
      }
  }


};


function bubbleSort(items) {
  const length = items.length;
  for(let i=0; i < (length-1); i++) {
      for(let j=0; j < (length-1); j++) {
          if(items[j] > items[j+1]) {
              [items[j], items[j+1]] = [items[j+1], items[j]];
          }
      }
  }
  return items;
}

function selectionSort(items) {
  const length = items.length;
  for(let start=0; start < length-1; start++) {
      let minIdx = start;
      let minVal = items[start];
      for(let i=start+1; i < length; i++) {
          if(items[i] < minVal) {
              minVal = items[i];
              minIdx = i;
          }
      }
      if(minIdx > start) {
          [items[start], items[minIdx]] = [items[minIdx],items[start]];
      }
  }
  return items;
}
