from lxml import etree
from zss import simple_distance, Node

html_table1 = "<html><body><table><thead><tr><td>(a) Geographical " \
              "grouping</td><td></td><td></td></tr></thead><tbody><tr><td>Groups i se I</td><td>Source of " \
              "Variation</td><td>Percentage Variation</td></tr><tr><td>GrouNorBha andOrisa ppulains</td><td>Among " \
              "groups</td><td>0.29</td></tr><tr><td>Grou2Soues\uff1aKakapul</td><td>Aongppulatisingroups</td><td>0.97" \
              "</td></tr><tr><td>Group3Southast:TaNaduPoplins</td><td>Wihinpulations</td><td>98.74</td></tr><tr><td>(" \
              "b) Linguisticgrouping</td><td></td><td></td></tr><tr><td>Groups in set 2</td><td>Source of " \
              "Variation</td><td>Percentage Variation </td></tr><tr><td>GroupIIdo-EurpeanOrisaand " \
              "hr</td><td>Amongroups</td><td>0.69</td></tr><tr><td>Group2Dravidan: Southern populatins</td><td>Among " \
              "poplatonsingroups</td><td>0.94</td></tr><tr><td></td><td>Within " \
              "populations</td><td>98.40</td></tr></tbody></table></body></html>"
html_table2 = "<html><body><table><thead><tr><td>(a) Geographical " \
              "grouping</td><td></td><td></td></tr></thead><tbody><tr><td>Groups i se I</td><td>Source of " \
              "Variation</td><td>Percentage Variation</td></tr><tr><td>GrouNorBha andOrisa ppulains</td><td>Among " \
              "groups</td><td>0.29</td></tr><tr><td>Grou2Soues\uff1aKakapul</td><td>Aongppulatisingroups</td><td>0.97" \
              "</td></tr><tr><td>Group3Southast:TaNaduPoplins</td><td>Wihinpulations</td><td>98.74</td></tr><tr><td>(" \
              "b) Linguisticgrouping</td><td></td><td></td></tr><tr><td>Groups in set 2</td><td>Source of " \
              "Variation</td><td>Percentage Variation </td></tr><tr><td>GroupIIdo-EurpeanOrisaand " \
              "hr</td><td>Amongroups</td><td>0.69</td></tr><tr><td>Group2Dravidan: Southern populatins</td><td>Among " \
              "poplatonsingroups</td><td>0.94</td></tr><tr><td></td><td>Within " \
              "populations</td><td>98.40</td></tr></tbody></table></body></html>"


def parse_html_table(html):
    parser = etree.HTMLParser()
    tree = etree.fromstring(html, parser)
    return tree


def html_tree_to_zss_tree(node):
    if node.tag is None:
        return Node(node.text)
    else:
        children = [html_tree_to_zss_tree(child) for child in node]
        return Node(node.tag, children)


def get_tree_size(zss_tree):
    if zss_tree is None:
        return 0
    size = 1
    for child in zss_tree.children:
        size += get_tree_size(child)
    return size


def tree_edit_distance(table1_html, table2_html):
    # Parse HTML tables into tree structures.
    tree1 = parse_html_table(table1_html)
    tree2 = parse_html_table(table2_html)

    # Convert HTML trees to zss trees.
    zss_tree1 = html_tree_to_zss_tree(tree1)
    zss_tree2 = html_tree_to_zss_tree(tree2)

    # Calculate the TED score according to the paper: https://arxiv.org/pdf/1911.10683.pdf

    max_dist = max(get_tree_size(zss_tree1), get_tree_size(zss_tree2))
    print(max_dist)

    ted_score = 1 - (simple_distance(zss_tree1, zss_tree2) / max_dist)

    return ted_score


ted_score = tree_edit_distance(html_table1, html_table2)
print(f"Tree Edit Distance Score: {ted_score}")
