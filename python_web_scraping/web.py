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

def display_interactive_graph(graph, output_file="web_graph_2.html"):
    """Create and save an interactive graph with clickable nodes."""
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
    node_text = []
    node_links = []

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(graph.nodes[node].get('label', node))
        node_links.append(node)  # Save the full URL for the node

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            size=20,
            color='#87CEEB',
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Interactive Site Reference Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)))

    # Add clickable links
    for i, url in enumerate(node_links):
        fig.add_annotation(
            x=node_x[i], y=node_y[i],
            text=f"<a href='{url}' target='_blank'>{node_text[i]}</a>",
            showarrow=False,
            font=dict(size=12, color="black"))

    fig.write_html(output_file)
    print(f"Graph saved as {output_file}. Open it in your browser to view.")

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

if __name__ == "__main__":
    start_url = input("Enter the URL of the web page: ")
    max_depth = int(input("Enter the crawling depth (maximum 10): "))
    web_graph = crawl_web(start_url, max_depth, max_links=20)
    display_interactive_graph(web_graph)

    while True:
        print("\nChoose an operation:")
        print("1. Perform BFS")
        print("2. Perform DFS")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")

        if choice == "1":
            bfs_start = input("Enter the start URL for BFS: ")
            if bfs_start in web_graph.nodes:
                bfs_result = bfs(web_graph, bfs_start)
                print("\nBFS Traversal Order:")
                for url in bfs_result:
                    print(url)
            else:
                print("The entered URL is not in the graph.")

        elif choice == "2":
            dfs_start = input("Enter the start URL for DFS: ")
            if dfs_start in web_graph.nodes:
                dfs_result = dfs(web_graph, dfs_start)
                print("\nDFS Traversal Order:")
                for url in dfs_result:
                    print(url)
            else:
                print("The entered URL is not in the graph.")

        elif choice == "3":
            print("Exiting the program.")
            break

        else:
            print("Invalid choice. Please try again.")
