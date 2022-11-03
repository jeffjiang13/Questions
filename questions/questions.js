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
