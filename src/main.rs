use std::io::Cursor;

use clap::Parser;
use image::{DynamicImage, GenericImageView, RgbImage};
use opencv::core::{Mat, Scalar, Vector};
use opencv::features2d::{AKAZE, draw_keypoints, Feature2DTrait};
use opencv::features2d::AKAZE_DescriptorType::DESCRIPTOR_MLDB;
use opencv::features2d::DrawMatchesFlags::DEFAULT;
use opencv::features2d::KAZE_DiffusivityType::DIFF_PM_G2;
use opencv::imgcodecs::{imdecode, imencode, IMREAD_COLOR, IMREAD_GRAYSCALE};
use opencv::imgproc::{THRESH_OTSU, threshold};
use opencv::types::VectorOfKeyPoint;
use rawler::imgop::develop::RawDevelop;

#[derive(clap::Subcommand, Clone, Debug)]
enum Mode {
    Composite {
        #[arg(short, long, value_delimiter = ',')]
        file: Vec<String>,

        #[arg(short, long)]
        output: String,
    },
    Test {
        #[arg(short, long, value_delimiter = ',')]
        file: Vec<String>,
    },
}

#[derive(clap::Parser)]
#[command(name = "mode", author, version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    mode: Mode,
}

fn main() {
    let args = Cli::parse();

    match args.mode {
        Mode::Composite { file, output } => {
            if file.len() > 2 {
                println!("Should specify more than 2 images.");
                return;
            }

            if output == "" {
                println!("Should specify output file.");
                return;
            }

            let first_image = convert_to_dynamic_image(&file[0]);
            let mut new_image: DynamicImage = first_image.clone();
            for f in &file[1..file.len()] {
                let image = convert_to_dynamic_image(f);
                new_image = lighten_composition_inner(&new_image, &image);
            }
            new_image.save(output).unwrap();
        }
        Mode::Test { file: files } => {
            if files.len() == 0 {
                println!("Should specify file.");
                return;
            }

            for f in files {
                draw_keypoints_from_filename(&f);
            }
        }
    };
}

fn draw_keypoints_from_filename(file: &str) {
    let image1 = convert_to_dynamic_image(file);
    let mat = dynamic_image_to_mat(&image1, IMREAD_COLOR);
    let vector = get_keypoints_from_filename(file);

    let mut output = Mat::default();
    draw_keypoints(&mat, &vector, &mut output, Scalar::all(-1f64), DEFAULT).unwrap();
    let output_image = mat_to_dynamic_image(&output);
    output_image.save("output.tiff").unwrap();
}

fn get_keypoints_from_filename(filename: &str) -> VectorOfKeyPoint {
    let image1 = convert_to_dynamic_image(&filename);
    let binarized_image1 = binarize(&image1);
    return get_keypoints(&binarized_image1);
}

fn get_keypoints(image: &DynamicImage) -> VectorOfKeyPoint {
    let mut akaze = AKAZE::create(DESCRIPTOR_MLDB, 0, 3, 0.001, 4, 4, DIFF_PM_G2).unwrap();
    let mut key_points = VectorOfKeyPoint::new();
    let mut descriptors = Mat::default();
    akaze
        .detect_and_compute(
            &dynamic_image_to_mat(image, IMREAD_GRAYSCALE),
            &[],
            &mut key_points,
            &mut descriptors,
            false,
        )
        .unwrap();
    return key_points;
}

/**
 * 画像を2値化する
 */
fn binarize(image: &DynamicImage) -> DynamicImage {
    let mat = dynamic_image_to_mat(image, IMREAD_GRAYSCALE);

    let max_thresh_val =
        threshold(&mat, &mut Mat::default(), 0.0, 255.0, THRESH_OTSU).unwrap() as f32;

    let (width, height) = image.dimensions();
    let mut new_image = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y);
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;
            let l = 0.299 * r + 0.587 * g + 0.114 * b;
            if l > max_thresh_val {
                new_image.put_pixel(x, y, image::Rgb([255, 255, 255]));
            } else {
                new_image.put_pixel(x, y, image::Rgb([0, 0, 0]));
            }
        }
    }
    return DynamicImage::from(new_image);
}

/**
 * DynamicImageをMatに変換する
 */
fn dynamic_image_to_mat(image: &DynamicImage, flags: i32) -> Mat {
    let mut bytes: Vec<u8> = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut bytes), image::ImageOutputFormat::Tiff)
        .unwrap();
    return imdecode(&bytes.as_slice(), flags).unwrap();
}

/**
* MatをDynamicImageに変換する
*/
fn mat_to_dynamic_image(mat: &Mat) -> DynamicImage {
    let mut buf = Vector::new();
    imencode(".tiff", &mat, &mut buf, &Vector::new()).unwrap();
    return image::load_from_memory(buf.as_slice()).unwrap();
}

fn convert_to_dynamic_image(file_path: &str) -> DynamicImage {
    let raw_image = rawler::decode_file(file_path).unwrap();
    let dev = RawDevelop::default();
    let image = dev
        .develop_intermediate(&raw_image)
        .unwrap()
        .to_dynamic_image()
        .unwrap();
    return image;
}

/**
 * 画像を比較明合成する
 */
fn lighten_composition_inner(image1: &DynamicImage, image2: &DynamicImage) -> DynamicImage {
    let (width, height) = image1.dimensions();
    let mut image = RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel1 = image1.get_pixel(x, y);
            let pixel2 = image2.get_pixel(x, y);

            let r1 = pixel1[0] as f32;
            let g1 = pixel1[1] as f32;
            let b1 = pixel1[2] as f32;

            let r2 = pixel2[0] as f32;
            let g2 = pixel2[1] as f32;
            let b2 = pixel2[2] as f32;

            let l1 = 0.299 * r1 + 0.587 * g1 + 0.114 * b1;
            let l2 = 0.299 * r2 + 0.587 * g2 + 0.114 * b2;

            if l1 > l2 {
                image.put_pixel(x, y, image::Rgb([r1 as u8, g1 as u8, b1 as u8]));
            } else {
                image.put_pixel(x, y, image::Rgb([r2 as u8, g2 as u8, b2 as u8]));
            }
        }
    }

    // return image;
    return DynamicImage::from(image);
}
