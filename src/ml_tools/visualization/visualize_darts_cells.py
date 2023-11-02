from search.darts_search.genotypes import Genotype
from graphviz import Digraph
import os

# Function to plot the architecture of a cell based on its genotype.
def plot(genotype, filename):
  # Create a directed graph with specific attributes for visualization.
  g = Digraph(
      format='png',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  # Add nodes for the input states of the cell.
  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  
  # Ensure the genotype has an even number of operations.
  assert len(genotype) % 2 == 0
  steps = len(genotype) // 2

  # Add nodes for intermediate states in the cell.
  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  # Add edges based on the operations and connections in the genotype.
  for i in range(steps):
    for k in [2*i, 2*i + 1]:
      op, j = genotype[k]
      if j == 0:
        u = "c_{k-2}"
      elif j == 1:
        u = "c_{k-1}"
      else:
        u = str(j-2)
      v = str(i)
      g.edge(u, v, label=op, fillcolor="gray")

  # Add node for the output state of the cell.
  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="gray")

  # Render the graph to a file and open it for viewing.
  g.render(filename, view=True)


def plot_darts_cell(genotype, save_path):
  # Here you define the genorype found by the NAS.
  # geno = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1), ('skip_connect', 3), ('max_pool_3x3', 1), ('skip_connect', 3), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

  # Plot the normal and reduction cells based on their genotypes.
  plot(genotype.normal, save_path + "/normal")
  plot(genotype.reduce, save_path + "/reduction")
