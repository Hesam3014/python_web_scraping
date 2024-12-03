import requests
from bs4 import BeautifulSoup
import networkx as nx
import plotly.graph_objects as go
from collections import deque
from urllib.parse import urlparse


def get_links(url, max_links=20):
    """Extract links from a web page, limited to a specified number."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=5, headers=headers)
        print(f"Response status: {response.status_code}")
        soup = BeautifulSoup(response.text, 'html.parser')
        links = set()
        for a_tag in soup.find_all('a', href=True)[:max_links]:
            href = a_tag['href']
            if href.startswith('http'):
                links.add(href)
        print(f"Extracted links: {len(links)}")
        return links
    except Exception as e:
        print(f"Error extracting links: {e}")
        return set()


def crawl_web(start_url, max_depth=10, max_links=20):
    """Crawl the web to extract links and construct a graph."""
    graph = nx.DiGraph()
    visited = set()
    queue = deque([(start_url, 0)])  # (URL, depth)

    while queue:
        current_url, depth = queue.popleft()
        if depth > max_depth or current_url in visited:
            continue

        visited.add(current_url)
        print(f"Processing: {current_url}, Depth: {depth}")
        links = get_links(current_url, max_links)
        graph.add_node(current_url, label=get_short_name(current_url))
        for link in links:
            graph.add_edge(current_url, link)
            queue.append((link, depth + 1))

    print(f"Total nodes: {graph.number_of_nodes()}")
    print(f"Total edges: {graph.number_of_edges()}")
    return graph


def get_short_name(url):
    """Extract a short name (domain) from a URL."""
    parsed_url = urlparse(url)
    return parsed_url.netloc


def bfs(graph, start_node):
    """Perform BFS on the graph starting from the specified node."""
    visited = set()
    queue = deque([start_node])
    bfs_order = []

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            bfs_order.append(node)
            queue.extend(graph.neighbors(node))  # Add all neighbors to the queue
    
    return bfs_order


def dfs(graph, start_node, visited=None, dfs_order=None):
    """Perform DFS on the graph starting from the specified node."""
    if visited is None:
        visited = set()
    if dfs_order is None:
        dfs_order = []

    visited.add(start_node)
    dfs_order.append(start_node)

    for neighbor in graph.neighbors(start_node):
        if neighbor not in visited:
            dfs(graph, neighbor, visited, dfs_order)
    
    return dfs_order


def display_colored_graph(graph, order, start_node, end_node, output_file, color):
    """Display the graph with BFS or DFS traversal highlighted."""
    pos = nx.spring_layout(graph, seed=42)
    edge_x = []
    edge_y = []

    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_color = []
    node_text = []

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(graph.nodes[node].get('label', node))

        # Set colors based on traversal order, start, and end nodes
        if node == start_node:
            node_color.append('yellow')
        elif node == end_node:
            node_color.append('red')
        elif node in order:
            node_color.append(color)
        else:
            node_color.append('#87CEEB')  # Default color

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            size=20,
            color=node_color,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'<br>{output_file.split(".")[0].capitalize()} Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)))

    fig.write_html(output_file)
    print(f"Graph saved as {output_file}. Open it in your browser to view.")


if __name__ == "__main__":
    start_url = input("Enter the start URL of the web page: ")
    max_depth = int(input("Enter the crawling depth (maximum 10): "))
    web_graph = crawl_web(start_url, max_depth, max_links=20)

    # Display discovered nodes and ask for the end node
    print("\nDiscovered Nodes:")
    for i, node in enumerate(web_graph.nodes()):
        print(f"{i + 1}: {node}")

    end_node_index = int(input("\nSelect the end node by number: ")) - 1
    end_node = list(web_graph.nodes())[end_node_index]

    # Perform BFS and DFS
    bfs_result = bfs(web_graph, start_url)
    dfs_result = dfs(web_graph, start_url)

    # Generate HTML files for each traversal
    display_colored_graph(web_graph, [], start_node=start_url, end_node=end_node, output_file="web_graph.html", color="#87CEEB")
    display_colored_graph(web_graph, bfs_result, start_node=start_url, end_node=end_node, output_file="bfs_graph.html", color="green")
    display_colored_graph(web_graph, dfs_result, start_node=start_url, end_node=end_node, output_file="dfs_graph.html", color="blue")
