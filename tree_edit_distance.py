import zss
import sqlparse

# def pretty_print(node, shift):
#     print(shift + str(node))
#     shift += '    '
#     for token in node.children:
#         pretty_print(token, shift + '    ')


class SqlNode:
    """
    Класс, предоставляющий интерфейс взаимодействия с библиотекой `zss` для синтаксических деревьев из библиотеки `sqlparse`
    """

    def __init__(self, node):
        self.children = []
        self.raw_node = node
        if type(node) == sqlparse.sql.Token or type(node) == sqlparse.sql.Identifier:
            self.label = str(node.value)
            return
        
        self.label = type(node).__name__
        for token in node.tokens:
            if token.is_whitespace:
                continue

            self.children.append(SqlNode(token))

    def __repr__(self):
        return str(type(self.raw_node)) + ' ' + self.label
    
    @staticmethod
    def get_children(self):
        return self.children
    
    @staticmethod
    def get_label(self):
        return self.label


def dist_comp(node1, node2):
    """
    Компаратор для двух вершин дерева
    """

    return int(node1 != node2)


def ratio(tree1 : SqlNode, tree2 : SqlNode):
    """
    Метрика Tree Edit Distance для двух деревьев

    Parameters
    ----------
    tree1 : SqlNode
        Корень первого дерева
    tree2 : SqlNode
        Корень второго дерева

    Return
    ------
        Значение метрики на отрезке [0;1]
    """

    edit_distance = zss.simple_distance(tree1, tree2, SqlNode.get_children, SqlNode.get_label, dist_comp)

    def __tree_nodes_count(root):
        cnt = 0
        for child in root.children:
            cnt += __tree_nodes_count(child)

        cnt += 1
        return cnt
    
    max_nodes = max(__tree_nodes_count(tree1), __tree_nodes_count(tree2))
    return max(1 - edit_distance/max_nodes, 0)


def parse_sql(query : str):
    return SqlNode(sqlparse.parse(query)[0])