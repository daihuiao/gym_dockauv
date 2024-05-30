class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(values):
    if not values:
        return None

    root_val = values.pop(0)
    if root_val is None:
        return None

    root = TreeNode(root_val)
    root.left = build_tree(values)
    root.right = build_tree(values)

    return root



class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        common_ancestor = None
        def find_ancestor(root,p,q):

            if p.val <= root.val and q.val >= root.val:
                nonlocal common_ancestor
                common_ancestor = root
                if root.left:
                    result_left = find_ancestor(root.left,p,q)
                if root.right:
                    result_right = find_ancestor(root.right,p,q)


        find_ancestor(root,p,q)
        return common_ancestor

solution = Solution()

root = TreeNode(6)
root.left = TreeNode(2)
root.right = TreeNode(8)
root.left.left = TreeNode(0)
root.left.right = TreeNode(4)
root.left.right.left = TreeNode(3)
root.left.right.right = TreeNode(5)
root.right.left = TreeNode(7)
root.right.right = TreeNode(9)
p = root.left
q = root.right

result = solution.lowestCommonAncestor(root, p, q)

