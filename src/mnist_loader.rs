pub mod mnist_loader {
  use mnist::*;
  pub fn get_mnist(train_len: u32, test_len: u32) -> (Vec<(Vec<f64>, Vec<f64>)>, Vec<(Vec<f64>, Vec<f64>)>) { // terrible terrible code but it gets the job done
    let Mnist {
      trn_img,
      trn_lbl,
      tst_img,
      tst_lbl,
      ..
    } = MnistBuilder::new()
      .label_format_one_hot()
      .training_set_length(train_len)
      .validation_set_length(test_len)
      .test_set_length(test_len)
      .finalize();
    let temp = trn_img.iter().map(|x| *x as f64/255.0).collect::<Vec<f64>>();
    let chunk_trn_img = temp.chunks(784).collect::<Vec<&[f64]>>();
    let temp = trn_lbl.iter().map(|x| *x as f64).collect::<Vec<f64>>();
    let chunk_trn_lbl = temp.chunks(10).collect::<Vec<&[f64]>>();
    let mut training_data = vec![];
    for i in 0..chunk_trn_img.len() {
      training_data.push((chunk_trn_img[i].to_vec(), chunk_trn_lbl[i].to_vec()));
    }
    let temp = tst_img.iter().map(|x| *x as f64/255.0).collect::<Vec<f64>>();
    let chunk_tst_img = temp.chunks(784).collect::<Vec<&[f64]>>();
    let temp = tst_lbl.iter().map(|x| *x as f64).collect::<Vec<f64>>();
    let chunk_tst_lbl = temp.chunks(10).collect::<Vec<&[f64]>>();
    let mut test_data = vec![];
    for i in 0..chunk_tst_img.len() {
      test_data.push((chunk_tst_img[i].to_vec(), chunk_tst_lbl[i].to_vec()));
    }
    (training_data, test_data)
  }
}