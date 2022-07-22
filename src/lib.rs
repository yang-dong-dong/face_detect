use opencv::{
    core::{Mat, Rect, Scalar, Size, Vector, CV_32F},
    dnn::{blob_from_image, prelude::NetTrait, read_net_from_caffe_buffer, Net},
    prelude::MatTraitConst,
};
use std::{error::Error, include_bytes};
pub struct FaceDetector {
    face_net: Net,
}
impl FaceDetector {
    pub fn new() -> Result<FaceDetector, Box<dyn Error>> {
        Ok(FaceDetector {
            face_net: read_net_from_caffe_buffer(
                &Vector::from_slice(include_bytes!("../assets/deploy.prototxt")),
                &Vector::from_slice(include_bytes!(
                    "../assets/res10_300x300_ssd_iter_140000.caffemodel"
                )),
            )?,
        })
    }
}
impl FaceDetector {
    pub fn detect_face(&mut self, image: &Mat) -> Result<Rect, Box<dyn Error>> {
        let face_blob = blob_from_image(
            &image,
            1.0,
            Size::new(300, 300),
            Scalar::new(104.0, 177.0, 123.0, 0.0),
            false,
            false,
            CV_32F,
        )?;
        self.face_net
            .set_input(&face_blob, "", 1.0, Scalar::default())?;
        let mut out_blob_names = Vector::new();
        let mut output_blobs: Vector<Mat> = Vector::new();
        out_blob_names.push("detection_out");
        output_blobs.push(Mat::default());
        self.face_net.forward(&mut output_blobs, &out_blob_names)?;
        let face = output_blobs.get(0)?;
        let x_left_bottom = (*face.at_nd::<f32>(&[0, 0, 0, 3])? * image.cols() as f32) as i32;
        let y_left_bottom = (*face.at_nd::<f32>(&[0, 0, 0, 4])? * image.rows() as f32) as i32;
        let x_right_top = (*face.at_nd::<f32>(&[0, 0, 0, 5])? * image.cols() as f32) as i32;
        let y_right_top = (*face.at_nd::<f32>(&[0, 0, 0, 6])? * image.rows() as f32) as i32;
        Ok(Rect::new(
            x_left_bottom,
            y_left_bottom,
            x_right_top - x_left_bottom,
            y_right_top - y_left_bottom,
        ))
    }
}
