# 目录

# 内容

### [450\. 删除二叉搜索树中的节点](https://leetcode-cn.com/problems/delete-node-in-a-bst/)


给定一个二叉搜索树的根节点 **root** 和一个值 **key**，删除二叉搜索树中的 **key **对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。

一般来说，删除节点可分为两个步骤：

1.  首先找到需要删除的节点；
2.  如果找到了，删除它。

**示例 1:**

![](https://tva1.sinaimg.cn/large/e6c9d24egy1h1xpqavawxj20xe08ywf0.jpg)

```
输入：root = [5,3,6,2,4,null,7], key = 3
输出：[5,4,6,2,null,null,7]
解释：给定需要删除的节点值是 3，所以我们首先找到 3 这个节点，然后删除它。
一个正确的答案是 [5,4,6,2,null,null,7], 如下图所示。
另一个正确答案是 [5,2,6,null,4,null,7]。

```

**示例 2:**

```
输入: root = [5,3,6,2,4,null,7], key = 0
输出: [5,3,6,2,4,null,7]
解释: 二叉树不包含值为 0 的节点
```

**示例 3:**

```
输入: root = [], key = 0
输出: []
```

**提示:**

*   节点数的范围 [0, 10<sup>4</sup>].
*   -10<sup>5</sup> <= Node.val <= 10<sup>5</sup>
*   节点值唯一
*   `root` 是合法的二叉搜索树
*   -10<sup>5</sup> <= key <= 10<sup>5</sup>

**进阶：** 要求算法时间复杂度为 O(h)，h 为树的高度。

#### 二叉搜索树的三个特性

-   二叉搜索树的中序遍历的序列是递增排序的序列

-   Successor代表的是中序遍历序列的下一个节点。即比当前节点大的最小节点，简称后继节点。 先取当前节点的右节点，然后一直取该节点的左节点，直到左节点为空，则最后指向的节点为后继节点。

    -   ```python
        def successor(root):
            root = root.right
            while root.left:
                root = root.left
            return root
        ```

-   Predecessor代表的是中序遍历序列的前一个节点。即比当前节点小的最大节点，简称前驱节点。先取当前节点的左节点，然后取该节点的右节点，直到右节点为空，则最后指向的节点为前驱节点。

    -   ```python
        def predecessor(root):
            root = root.left
            while root.right:
                root = root.right
            return root


#### Solution

```python
class Solution:
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if not root:
            return None
        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:
            if not root.left and not root.right:
                return None
            if not root.left or not root.right:
                return root.left if root.left else root.right
            # 在右子树寻找后继节点child
            
            parent, child = root, root.right
            while child.left:
                parent = child
                child = child.left
            root.val = child.val
            if child == parent.left:
                parent.left = self.deleteNode(child, child.val)
            elif child == parent.right:
                parent.right = self.deleteNode(child, child.val)
        return root
```

