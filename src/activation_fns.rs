pub mod activation_fns {
  use crate::network::network::Node;

  pub fn sigmoid(nodes: &mut [Node]) {
    let act = |x: f64| {
      if x > 20.0 {
        1.0
      } else if x < -20.0 {
        0.0
      } else {
        1.0/(1.0 + f64::exp(-x))
      }
    };
    for i in 0..nodes.len() {
      let z = nodes[i].activation;
      nodes[i].activation = act(z);
      nodes[i].as_respect_z = (0..nodes.len())
        .map(|j| if j == i { act(z)*(1.0 - act(z)) } else { 0.0 }).collect();
    }
  }

  pub fn leaky_relu(nodes: &mut [Node]) {
    let leak = 0.01;
    for i in 0..nodes.len() {
      let z = nodes[i].activation;
      nodes[i].activation = if z > 0.0 { z } else { leak * z };
      nodes[i].as_respect_z = (0..nodes.len())
        .map(|j| if j == i { if z > 0.0 { 1.0 } else { leak } } else { 0.0 }).collect();
    }
  }

  pub fn soft_max(nodes: &mut [Node]) {
    let max = nodes.iter().map(|node| node.activation).fold(f64::NEG_INFINITY, |acc, e| acc.max(e));
    let exps = nodes.iter().map(|node| f64::exp(node.activation - max)).collect::<Vec<f64>>();
    let total = exps.iter().sum::<f64>();
    for i in 0..nodes.len() {
      nodes[i].activation = exps[i]/total;
    }
    let zs = nodes.iter().map(|node| node.activation).collect::<Vec<f64>>();
    for i in 0..nodes.len() {
      nodes[i].as_respect_z = (0..nodes.len())
        .map(|j| if j == i { nodes[i].activation*(1.0 - nodes[i].activation) } else { -nodes[i].activation*nodes[j].activation }).collect();
    }
    
  }
}