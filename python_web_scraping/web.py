import requests
from bs4 import BeautifulSoup
import networkx as nx
import plotly.graph_objects as go
from collections import deque
from urllib.parse import urlparse
import json
import os

from urllib.parse import urlparse

def get_links(url, max_links=20):
    """Extract links from a web page, limited to a specified number."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=5, headers=headers)
        print(f"Response status: {response.status_code}")
        soup = BeautifulSoup(response.text, 'html.parser')

        # Parse the starting domain
        parsed_start_url = urlparse(url)
        start_domain = parsed_start_url.netloc

        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith('http'):  # Full URL
                parsed_href = urlparse(href)
                if parsed_href.netloc != start_domain:  # Exclude same-domain links
                    links.add(href)
            elif href.startswith('/'):  # Relative URL
                full_url = f"{parsed_start_url.scheme}://{start_domain}{href}"
                parsed_full_url = urlparse(full_url)
                if parsed_full_url.netloc != start_domain:  # Exclude same-domain links
                    links.add(full_url)

            if len(links) >= max_links:  # Stop if we reach the max_links limit
                break

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


def save_graph_to_json(graph, file_path):
    """Save the graph nodes and edges to a JSON file."""
    data = {
        "nodes": list(graph.nodes(data=True)),
        "edges": list(graph.edges())
    }
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Graph saved to {file_path}.")


def load_graph_from_json(file_path):
    """Load the graph nodes and edges from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    graph = nx.DiGraph()
    for node, attrs in data["nodes"]:
        graph.add_node(node, **attrs)
    for edge in data["edges"]:
        graph.add_edge(*edge)
    print(f"Graph loaded from {file_path}.")
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


def topological_sort(graph):
    """Perform topological sorting on the graph if it's a DAG."""
    try:
        order = list(nx.topological_sort(graph))
        print("\nTopological Sort Order:")
        for i, node in enumerate(order):
            print(f"{i + 1}: {node}")
        return order
    except nx.NetworkXUnfeasible:
        print("\nThe graph contains cycles and is not a DAG. Topological sorting is not possible.")
        return None


def kosaraju(graph):
    """Perform Kosaraju's algorithm to find strongly connected components (SCCs)."""
    # Step 1: DFS on original graph to get finishing times
    def dfs_first_pass(node, visited, stack):
        visited.add(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                dfs_first_pass(neighbor, visited, stack)
        stack.append(node)  # Push the node to stack after visiting all neighbors

    stack = []
    visited = set()
    for node in graph.nodes():
        if node not in visited:
            dfs_first_pass(node, visited, stack)

    # Step 2: Reverse the graph
    reversed_graph = graph.reverse()

    # Step 3: DFS on reversed graph in order of finishing times
    def dfs_second_pass(node, visited, scc):
        visited.add(node)
        scc.append(node)
        for neighbor in reversed_graph.neighbors(node):
            if neighbor not in visited:
                dfs_second_pass(neighbor, visited, scc)

    visited.clear()
    sccs = []
    while stack:
        node = stack.pop()
        if node not in visited:
            scc = []
            dfs_second_pass(node, visited, scc)
            sccs.append(scc)

    return sccs


def display_colored_graph_with_scc(graph, order, start_node, end_node, output_file, color, sccs=None):
    """Display the graph with BFS, DFS, or Kosaraju's SCCs highlighted."""
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

    # If SCCs are provided, color them differently
    scc_colors = {}
    if sccs:
        for idx, scc in enumerate(sccs):
            for node in scc:
                scc_colors[node] = f"rgb({(idx * 50) % 255}, {(idx * 100) % 255}, 255)"

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(graph.nodes[node].get('label', node))

        # Set colors based on traversal order, start, end nodes, or SCCs
        if node == start_node:
            node_color.append('yellow')
        elif node == end_node:
            node_color.append('red')
        elif node in order:
            node_color.append(color)
        elif node in scc_colors:
            node_color.append(scc_colors[node])
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
    folder_path = os.path.dirname(os.path.abspath(__file__))  # Save files in script's directory

    start_url = input("Enter the start URL of the web page: ")
    max_depth = int(input("Enter crawling depth (maximum 10): "))
    json_file = os.path.join(folder_path, "web_graph.json")

    # Crawl and save the graph
    web_graph = crawl_web(start_url, max_depth, max_links=20)
    save_graph_to_json(web_graph, json_file)

    # Load the graph from JSON
    web_graph = load_graph_from_json(json_file)

    # Display discovered nodes and ask for the end node
    print("\nDiscovered Nodes:")
    for i, node in enumerate(web_graph.nodes()):
        print(f"{i + 1}: {node}")

    end_node_index = int(input("\nSelect the end node by number: ")) - 1
    end_node = list(web_graph.nodes())[end_node_index]

    # Perform BFS and DFS
    bfs_result = bfs(web_graph, start_url)
    dfs_result = dfs(web_graph, start_url)

    # Perform Kosaraju's algorithm
    kosaraju_sccs = kosaraju(web_graph)

    # Perform Topological Sorting
    topo_sort_result = topological_sort(web_graph)

    # Save topological sort order to a file if it exists
    if topo_sort_result:
        topo_file = os.path.join(folder_path, "topological_order.txt")
        with open(topo_file, 'w') as f:
            f.write("\n".join(topo_sort_result))
        print(f"\nTopological order saved to {topo_file}.")

    # Generate HTML files for each traversal
    display_colored_graph_with_scc(web_graph, [], start_node=start_url, end_node=end_node,
                                   output_file=os.path.join(folder_path, "web_graph.html"), color="#87CEEB")
    display_colored_graph_with_scc(web_graph, bfs_result, start_node=start_url, end_node=end_node,
                                   output_file=os.path.join(folder_path, "bfs_graph.html"), color="green")
    display_colored_graph_with_scc(web_graph, dfs_result, start_node=start_url, end_node=end_node,
                                   output_file=os.path.join(folder_path, "dfs_graph.html"), color="blue")
    display_colored_graph_with_scc(web_graph, [], start_node=start_url, end_node=end_node,
                                   output_file=os.path.join(folder_path, "kosaraju_graph.html"), color="purple", sccs=kosaraju_sccs)
