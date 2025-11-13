#!/usr/bin/env python3
"""
Visualize BehaviorTree.CPP XML format using Graphviz
"""
import xml.etree.ElementTree as ET
import subprocess
import sys

def parse_bt_node(node, graph, parent_id=None, node_counter=[0]):
    """Recursively parse BT XML and build graph"""
    node_counter[0] += 1
    current_id = f"node_{node_counter[0]}"
    
    # Get node type and name
    tag = node.tag
    name = node.get('name', node.get('ID', tag))
    
    # Color coding by node type
    colors = {
        'Sequence': 'lightblue',
        'Selector': 'lightgreen',
        'Parallel': 'lightyellow',
        'Inverter': 'lightcoral',
        'Action': 'lightgray'
    }
    color = colors.get(tag, 'white')
    
    # Add node to graph
    shape = 'box' if tag == 'Action' else 'ellipse'
    graph.append(f'  {current_id} [label="{name}\\n({tag})", fillcolor="{color}", style=filled, shape={shape}];')
    
    # Connect to parent
    if parent_id:
        graph.append(f'  {parent_id} -> {current_id};')
    
    # Process children
    for child in node:
        parse_bt_node(child, graph, current_id, node_counter)
    
    return current_id

def visualize_bt_xml(xml_file):
    """Convert BT XML to DOT and PNG"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Find the main behavior tree
        bt = root.find('.//BehaviorTree')
        if bt is None:
            print("Error: No BehaviorTree found in XML")
            return False
        
        # Build DOT graph
        graph_lines = ['digraph BehaviorTree {']
        graph_lines.append('  rankdir=TB;')
        graph_lines.append('  node [fontname="Arial"];')
        
        # Parse the tree starting from first child of BehaviorTree
        first_node = list(bt)[0]
        parse_bt_node(first_node, graph_lines)
        
        graph_lines.append('}')
        
        # Write DOT file
        dot_content = '\n'.join(graph_lines)
        dot_file = xml_file.replace('.xml', '.dot')
        png_file = xml_file.replace('.xml', '.png')
        
        with open(dot_file, 'w') as f:
            f.write(dot_content)
        
        print(f"✓ Generated {dot_file}")
        
        # Convert to PNG using graphviz
        try:
            subprocess.run(['dot', '-Tpng', dot_file, '-o', png_file], check=True)
            print(f"✓ Generated {png_file}")
            print(f"\nOpen the PNG file to see your behavior tree visualization!")
            return True
        except subprocess.CalledProcessError:
            print("Error: Failed to generate PNG. Is graphviz installed?")
            print("Install with: sudo apt install graphviz")
            return False
        except FileNotFoundError:
            print("Error: 'dot' command not found. Please install graphviz:")
            print("  sudo apt install graphviz")
            return False
            
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return False
    except FileNotFoundError:
        print(f"Error: File '{xml_file}' not found")
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        xml_file = 'piling_safety_trees.xml'
        print(f"Usage: {sys.argv[0]} <xml_file>")
        print(f"Using default: {xml_file}\n")
    else:
        xml_file = sys.argv[1]
    
    visualize_bt_xml(xml_file)