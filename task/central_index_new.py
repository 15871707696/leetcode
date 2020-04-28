# 输入:
# nums = [1, 7, 3, 6, 5, 6]
# 输出: 3
# 解释:
# 索引3 (nums[3] = 6) 的左侧数之和(1 + 7 + 3 = 11)，与右侧数之和(5 + 6 = 11)相等。
# 同时, 3 也是第一个符合要求的中心索引。
class A():
    def pivotIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums_len = len(nums)
        sum_nums = sum(nums)
        sum_left = 0
        for i in range(nums_len):
            if i >= 1:
                sum_left += nums[i - 1]
            sum_right = sum_nums - sum_left - nums[i]
            if sum_left == sum_right:
                return i, nums[i]
        return -1, -1



if __name__ == '__main__':
    nums = [1, 7, 3, 6, 5, 6]
    index, central = A().pivotIndex(nums)
