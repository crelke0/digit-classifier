mod network;
use crate::network::network::*;
mod mnist_loader;
use crate::mnist_loader::mnist_loader::get_mnist;
mod functions;
use functions::functions::*;
use std::fs;
use png;
use std::fs::File;

fn main() {
  let train = false;
  let data = get_mnist(50_000, 10_000);
  let mut training_data = data.0;
  let test_data = data.1;
  if train {
    println!("starting training...");
    let mut network = Network::new(&[784, 100, 80, 10], &[ActivationFn::Sigmoid, ActivationFn::Sigmoid, ActivationFn::SoftMax]);
    network.train(&mut training_data, &test_data, 10, 100, 0.25, 0.002, Cost::CrossEntropy, true, true);
  } else {
    let contents = fs::read_to_string("pre_computed/net.json").expect("Unable to read file");
    let mut network = serde_json::from_str::<Network>(&contents).expect("Unable to deserialize file");

    let decoder = png::Decoder::new(File::open("input/number.png").unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let bytes = &buf[..info.buffer_size()].iter().map(|x| 1.0 - *x as f64/255.0).collect::<Vec<f64>>();
    let padding = ((28.0 - (bytes.len() as f64 / 4.0).sqrt()) / 2.0) as usize;

    let mut input = vec![0.0; 28*padding];
    for i in 0..(28-padding*2) {
      input.append(&mut vec![0.0; padding]);
      for j in 0..(28-padding*2) {
        input.push(bytes[(i*(28-padding*2)+j)*4]);
      }
      input.append(&mut vec![0.0; padding]);
    }
    input.append(&mut vec![0.0; 28*padding]);
    let probabilities = network.run(&input);
    let mut guess = 0;
    let mut highest = 0.0; 
    for i in 0..probabilities.len() {
      if probabilities[i] > highest {
        highest = probabilities[i];
        guess = i;
      }
    }
    println!("Probabilities: {:?} \nGuess: {}", probabilities, guess);
  }
}