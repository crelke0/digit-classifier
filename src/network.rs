pub mod network {
  use rand::prelude::*;
  pub struct Node {
    weights: Vec<f64>,
    bias: f64,
    d_weights: Vec<f64>,
    d_bias: f64,
    pub activation: f64, 
    pub as_respect_z: Vec<f64>
  }
  impl Node {
    fn new(num_weights: usize) -> Node {
      let mut rng = rand::thread_rng();
      Node {
        weights: (0..num_weights).map(|_| rng.gen::<f64>()*2.0-1.0).collect(),
        bias: 0.0,
        d_weights: vec![0.0; num_weights],
        d_bias: 0.0,
        activation: 0.0,
        as_respect_z: vec![]
      }
    }
    fn update_activation(&mut self, previous_activations: &[f64]) {
      self.activation = self.bias;
      for i in 0..previous_activations.len() {
        self.activation += previous_activations[i]*self.weights[i]
      }
    }
    fn add_change(&mut self, c_respect_z: f64, previous_activations: &[f64]) {
      self.d_bias += c_respect_z;
      for i in 0..self.weights.len() {
        self.d_weights[i] += c_respect_z*previous_activations[i];
      }
    }
    fn apply_change(&mut self, coef: f64, lambda_over_size: f64) {
      self.bias -= self.d_bias*coef;
      self.d_bias = 0.0;
      for i in 0..self.weights.len() {
        self.weights[i] -= self.d_weights[i]*coef + coef*lambda_over_size*self.weights[i];
        self.d_weights[i] = 0.0
      }
    }
  }
  
  trait Layer {
    fn propagate(&mut self, c_respect_as: &[f64]);
    fn activations(&self) -> Vec<f64>;
    fn feed(&mut self, input: &[f64]);
    fn num_nodes(&self) -> usize {
      self.activations().len()
    }
    fn apply_changes(&mut self, coef: f64, lambda_over_size: f64);
  }
  
  struct HiddenLayer { // HiddenLayer is actually for hidden layers AND the output layer, since they act the same
    nodes: Vec<Node>,
    previous_layer: Box<dyn Layer>,
    activation_fn: fn(&mut [Node])
  }
  impl HiddenLayer {
    fn new(previous_layer: Box<dyn Layer>, node_amount: usize, activation_fn: fn(&mut [Node])) -> HiddenLayer {
      let mut nodes = vec![];
      for _ in 0..node_amount {
        nodes.push(Node::new(previous_layer.num_nodes()));
      }
      HiddenLayer {
        nodes: nodes,
        previous_layer: previous_layer,
        activation_fn: activation_fn
      }
    }
  }
  impl Layer for HiddenLayer {
    fn activations(&self) -> Vec<f64> {
      self.nodes.iter().map(|x| x.activation).collect()
    }
    fn feed(&mut self, input: &[f64]) {
      self.previous_layer.feed(input);
      for node in &mut self.nodes {
        node.update_activation(&self.previous_layer.activations())
      }
      (self.activation_fn)(&mut self.nodes)
    }
    fn propagate(&mut self, c_respect_as: &[f64]) {
      let mut c_respect_zs = vec![];
      for node in &self.nodes {
        let mut c_respect_z = 0.0;
        for i in 0..node.as_respect_z.len() {
          c_respect_z += node.as_respect_z[i]*c_respect_as[i];
        }
        
        c_respect_zs.push(c_respect_z);
      }
      for i in 0..self.num_nodes() {
        self.nodes[i].add_change(c_respect_zs[i], &self.previous_layer.activations());
      }
      let mut new_c_respect_as = vec![0.0; self.previous_layer.num_nodes()];
      for j in 0..self.num_nodes() {
        for k in 0..new_c_respect_as.len() {
          new_c_respect_as[k] += self.nodes[j].weights[k]*c_respect_zs[j];
        }
      }
  
      self.previous_layer.propagate(&new_c_respect_as);
    }
    fn apply_changes(&mut self, coef: f64, lambda_over_size: f64) {
      for node in &mut self.nodes {
        node.apply_change(coef, lambda_over_size)
      }
      self.previous_layer.apply_changes(coef, lambda_over_size);
    }
  }
  
  struct InputLayer {
    activations: Vec<f64>
  }
  impl InputLayer {
    fn new(size: usize) -> InputLayer{
      InputLayer { activations: vec![0.0; size] }
    }
  }
  impl Layer for InputLayer {
    fn activations(&self) -> Vec<f64> {
      self.activations.clone()
    }
    fn feed(&mut self, input: &[f64]) {
      self.activations = input.to_vec();
    }
    fn apply_changes(&mut self, _coef: f64, _lambda_over_size: f64) { }
    fn propagate(&mut self, _c_respect_as: &[f64]) { }
  }
  
  pub struct Network {
    output_layer: HiddenLayer
  }
  
  impl Network {
    pub fn new(dim: &[usize], activation_fns: &[fn(&mut [Node])]) -> Network {
      let mut output_layer = HiddenLayer::new(Box::new(InputLayer::new(dim[0])), dim[1], activation_fns[0]);
      for i in 1..dim.len() {
        output_layer = HiddenLayer::new(Box::new(output_layer), dim[i], activation_fns[i]);
      }
      Network {
        output_layer: output_layer
      }
    }
    fn output(&self) -> Vec<f64>{
      self.output_layer.activations()
    }
    fn test(&mut self, test_set: &[(Vec<f64>, Vec<f64>)]) {
      let mut total_correct = 0;
      for (inp, out) in test_set {
        self.output_layer.feed(inp);
        let mut max = 0f64;
        let mut index = 0;
        let prediction = self.output();
        for i in 0..prediction.len() {
          if prediction[i] > max {
            max = prediction[i];
            index = i;
          }
        }
        if out[index] == 1.0 {
          total_correct += 1;
        }
      }
      println!("{} / {} = {}%", total_correct, test_set.len(), total_correct as f64 / test_set.len() as f64 * 100.0);
    }
    fn train_batch(&mut self, batch: &[(Vec<f64>, Vec<f64>)], learning_rate: f64, cost_der: fn((&f64, &f64)) -> f64, lambda: f64) {
      for (inp, out) in batch {
        self.output_layer.feed(inp);
        let c_respect_as = self.output().iter().zip(out).map(cost_der).collect::<Vec<f64>>();
        self.output_layer.propagate(&c_respect_as);
      }
      self.output_layer.apply_changes(learning_rate/batch.len() as f64, lambda / batch.len() as f64);
    }
    pub fn train(&mut self, training_set: &mut [(Vec<f64>, Vec<f64>)], test_set: &[(Vec<f64>, Vec<f64>)], batch_size: usize, 
            epochs: usize, learning_rate: f64, lambda: f64, cost_der: fn((&f64, &f64)) -> f64, test_epochs: bool) {
      let mut rng = rand::thread_rng();
      for epoch in 1..epochs+1 {
        training_set.shuffle(&mut rng);
        for batch in training_set.chunks(batch_size) {
          self.train_batch(batch, learning_rate, cost_der, lambda);
        }
        if test_epochs {
          print!("epoch {}: ", epoch);
          self.test(test_set)
        }
      }
      if !test_epochs {
        self.test(test_set);
      }
    }
  }

  pub fn quadratic_cost_der((a, y): (&f64, &f64)) -> f64 {
    2.0*(a - y)
  }
  pub fn crossentropy_cost_der((a, y): (&f64, &f64)) -> f64 {
    -y/a + (1.0 - y)/(1.0 - a)
  }
}
