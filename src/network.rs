pub mod network {
  use rand::prelude::*;
  struct Node {
    weights: Vec<f32>,
    bias: f32,
    d_weights: Vec<f32>,
    d_bias: f32,
    activation_fn: fn(f32) -> f32,
    activation_fn_der: fn(f32) -> f32,
    activation: (f32, f32) // (actual activation, activation before activation function)
  }
  impl Node {
    fn new(num_weights: usize, activation_fn: fn(f32) -> f32, activation_fn_der: fn(f32) -> f32) -> Node {
      let mut rng = rand::thread_rng();
      Node {
        weights: (0..num_weights).map(|_| rng.gen::<f32>()*2.0-1.0).collect(),
        bias: rng.gen::<f32>()*2.0-1.0,
        d_weights: vec![0.0; num_weights],
        d_bias: 0.0,
        activation_fn: activation_fn,
        activation_fn_der: activation_fn_der,
        activation: (0.0, 0.0)
      }
    }
    fn update_activation(&mut self, previous_activations: &[(f32, f32)]) {
      let mut activation = self.bias;
      for i in 0..previous_activations.len() {
        activation += previous_activations[i].0*self.weights[i]
      }
      self.activation = ((self.activation_fn)(activation), activation)
    }
    fn add_change(&mut self, c_respect_z: f32, previous_activations: &[(f32, f32)]) {
      self.d_bias -= c_respect_z;
      for i in 0..self.weights.len() {
        self.d_weights[i] -= c_respect_z*previous_activations[i].0;
      }
    }
    fn apply_change(&mut self, coef: f32) {
      self.bias += self.d_bias*coef;
      self.d_bias = 0.0;
      for i in 0..self.weights.len() {
        self.weights[i] += self.d_weights[i]*coef;
        self.d_weights[i] = 0.0
      }
    }
  }
  
  trait Layer {
    fn propagate(&mut self, c_respect_as: &[f32]);
    fn activations(&self) -> Vec<(f32, f32)>;
    fn feed(&mut self, input: &[f32]);
    fn num_nodes(&self) -> usize {
      self.activations().len()
    }
    fn apply_changes(&mut self, coef: f32);
  }
  
  struct HiddenLayer { // HiddenLayer is actually for hidden layers AND the output layer, since they act the same
    nodes: Vec<Node>,
    previous_layer: Box<dyn Layer>
  }
  impl HiddenLayer {
    fn new(previous_layer: Box<dyn Layer>, node_amount: usize, activation_fn: fn(f32) -> f32, activation_fn_der: fn(f32) -> f32) -> HiddenLayer {
      let mut nodes = vec![];
      for _ in 0..node_amount {
        nodes.push(Node::new(previous_layer.num_nodes(), activation_fn, activation_fn_der))
      }
      HiddenLayer {
        nodes: nodes,
        previous_layer: previous_layer
      }
    }
  }
  impl Layer for HiddenLayer {
    fn activations(&self) -> Vec<(f32, f32)> {
      self.nodes.iter().map(|x| x.activation).collect()
    }
    fn feed(&mut self, input: &[f32]) {
      self.previous_layer.feed(input);
      for node in &mut self.nodes {
        node.update_activation(&self.previous_layer.activations())
      }
    }
    fn propagate(&mut self, c_respect_as: &[f32]) {
      let mut c_respect_zs = vec![];
      for i in 0..self.num_nodes() {
        c_respect_zs.push(c_respect_as[i]*(self.nodes[i].activation_fn_der)(self.activations()[i].1));
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
    fn apply_changes(&mut self, coef: f32) {
      for node in &mut self.nodes {
        node.apply_change(coef)
      }
      self.previous_layer.apply_changes(coef);
    }
  }
  
  struct InputLayer {
    activations: Vec<f32>
  }
  impl InputLayer {
    fn new(size: usize) -> InputLayer{
      InputLayer { activations: vec![0.0; size] }
    }
  }
  impl Layer for InputLayer {
    fn activations(&self) -> Vec<(f32, f32)> {
      self.activations.iter().map(|x| (*x, *x)).collect()
    }
    fn feed(&mut self, input: &[f32]) {
      self.activations = input.to_vec();
    }
    fn apply_changes(&mut self, _coef: f32) { }
    fn propagate(&mut self, _c_respect_as: &[f32]) { }
  }
  
  pub struct Network {
    output_layer: HiddenLayer
  }
  
  impl Network {
    pub fn new(dim: &[usize], activation_fn: fn(f32) -> f32, activation_fn_der: fn(f32) -> f32) -> Network {
      let mut output_layer = HiddenLayer::new(Box::new(InputLayer::new(dim[0])), dim[1], activation_fn, activation_fn_der);
      for i in 1..dim.len() {
        output_layer = HiddenLayer::new(Box::new(output_layer), dim[i], activation_fn, activation_fn_der)
      }
      Network {
        output_layer: output_layer
      }
    }
    fn output(&self) -> Vec<f32>{
      self.output_layer.activations().iter().map(|x| x.0).collect()
    }
    fn test(&mut self, test_set: &[(Vec<f32>, Vec<f32>)]) {
      let mut total_correct = 0;
      for (inp, out) in test_set {
        self.output_layer.feed(inp);
        let mut max = 0f32;
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
      println!("{} / {} = {}%", total_correct, test_set.len(), total_correct as f32 / test_set.len() as f32 * 100.0);
    }
    fn train_batch(&mut self, batch: &[(Vec<f32>, Vec<f32>)], training_speed: f32, cost_der: fn((&f32, &f32)) -> f32) {
      for (inp, out) in batch {
        self.output_layer.feed(inp);
        let c_respect_as = self.output().iter().zip(out).map(cost_der).collect::<Vec<f32>>();
        self.output_layer.propagate(&c_respect_as);
      }
      self.output_layer.apply_changes(training_speed/batch.len() as f32);
    }
    pub fn train(&mut self, training_set: &mut [(Vec<f32>, Vec<f32>)], test_set: &[(Vec<f32>, Vec<f32>)], batch_size: usize, 
            epochs: usize, training_speed: f32, cost_der: fn((&f32, &f32)) -> f32, test_epochs: bool) {
      let mut rng = rand::thread_rng();
      for epoch in 1..epochs+1 {
        training_set.shuffle(&mut rng);
        for batch in training_set.chunks(batch_size) {
          self.train_batch(batch, training_speed, cost_der);
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
  
  pub fn sigmoid(x: f32) -> f32 {
    if x > 20.0 {
      1.0
    } else if x < -20.0 {
      0.0
    } else {
      1.0/(1.0 + f32::exp(-x))
    }
  }
  pub fn sigmoid_der(x: f32) -> f32 {
    sigmoid(x)*(1.0 - sigmoid(x))
  }
  
  pub fn leaky_relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.01*x }
  }
  pub fn leaky_relu_der(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.01 }
  }
  
  pub fn quadratic_cost_der((a, y): (&f32, &f32)) -> f32{
    2.0*(a - y)
  }
  pub fn crossentropy_cost_der((a, y): (&f32, &f32)) -> f32{
    -a/y + (1.0 - y)/(1.0 - a)
  }
}
