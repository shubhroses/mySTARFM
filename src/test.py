i, j = 0, len(nums)-1

res = 0
while i <= j:
    res += int(str(nums[i]) + str(nums[j]))
    i += 1
    j -= 1
return res