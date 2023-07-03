pub mod activation_fns {
  use crate::network::network::Node;

  pub fn sigmoid(nodes: &mut [Node]) {
    let act = |x: f32| {
      if x > 20.0 {
        1.0
      } else if x < -20.0 {
        0.0
      } else {
        1.0/(1.0 + f32::exp(-x))
      }
    };
    for node in nodes {
      let z = node.activation;
      node.activation = act(z);
      node.a_respect_z = act(z)*(1.0 - act(z));
    }
  }

  fn leaky_relu(nodes: &mut [Node]) {
    let leak = 0.01;
    for node in nodes {
      let z = node.activation;
      node.activation = if z > 0.0 { z } else { leak * z };
      node.a_respect_z = if z > 0.0 { 1.0 } else { leak }
    }
  }

  pub fn soft_max(nodes: &mut [Node]) {
    let exps = nodes.iter().map(|node| f32::exp(node.activation)).collect::<Vec<f32>>();
    let total = exps.iter().sum::<f32>();
    for i in 0..nodes.len() {
      nodes[i].activation = exps[i]/total;
      let remaining = total - exps[i];
      nodes[i].a_respect_z = remaining*exps[i]/((exps[i] + remaining).powf(2.0));
    }
  }
}