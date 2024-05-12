use image::DynamicImage;
use opencv::core::{DMatch, KeyPoint, Mat, Scalar, Vector};
use opencv::features2d::draw_matches;
use opencv::features2d::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS;
use opencv::imgcodecs::{imdecode, imencode};
use rawler::imgop::develop::RawDevelop;
use std::io::Cursor;

pub fn draw_match_points(
    k1: &Vector<KeyPoint>,
    k2: &Vector<KeyPoint>,
    matches: &Vector<DMatch>,
    mat1: &Mat,
    mat2: &Mat,
) {
    let mut output = Mat::default();
    draw_matches(
        &mat1,
        &k1,
        &mat2,
        &k2,
        &matches,
        &mut output,
        Scalar::all(-1f64),
        Scalar::all(-1f64),
        &Vector::new(),
        NOT_DRAW_SINGLE_POINTS,
    )
    .unwrap();
    mat_to_dynamic_image(&output).save("matches.tiff").unwrap();
}

/**
 * DynamicImageをMatに変換する
 */
pub fn dynamic_image_to_mat(image: &DynamicImage, flags: i32) -> Mat {
    let mut bytes: Vec<u8> = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut bytes), image::ImageOutputFormat::Tiff)
        .unwrap();
    return imdecode(&bytes.as_slice(), flags).unwrap();
}

/**
 * MatをDynamicImageに変換する
 */
pub fn mat_to_dynamic_image(mat: &Mat) -> DynamicImage {
    let mut buf = Vector::new();
    imencode(".tiff", &mat, &mut buf, &Vector::new()).unwrap();
    return image::load_from_memory(buf.as_slice()).unwrap();
}

pub fn convert_to_dynamic_image(file_path: &str) -> DynamicImage {
    let file_ext = file_path.split(".").last().unwrap();

    match file_ext {
        "CR3" => {
            let raw_image = rawler::decode_file(file_path).unwrap();
            let dev = RawDevelop::default();
            let image = dev
                .develop_intermediate(&raw_image)
                .unwrap()
                .to_dynamic_image()
                .unwrap();
            return image;
        }
        "tiff" => {
            return image::open(file_path).unwrap();
        }
        _ => {
            panic!("Unsupported file type.")
        }
    }
}
