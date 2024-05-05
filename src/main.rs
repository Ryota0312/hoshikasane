use std::io::Cursor;

use clap::Parser;
use image::{DynamicImage, GenericImageView, RgbImage};
use opencv::core::{DMatch, KeyPoint, Mat, no_array, NORM_HAMMING, Scalar, Vector};
use opencv::features2d::{AKAZE, BFMatcher, DescriptorMatcherTrait, DescriptorMatcherTraitConst, draw_keypoints, draw_matches, Feature2DTrait};
use opencv::features2d::AKAZE_DescriptorType::{DESCRIPTOR_KAZE, DESCRIPTOR_MLDB};
use opencv::features2d::DrawMatchesFlags::{DEFAULT, NOT_DRAW_SINGLE_POINTS};
use opencv::features2d::KAZE_DiffusivityType::DIFF_PM_G2;
use opencv::imgcodecs::{imdecode, imencode, IMREAD_COLOR, IMREAD_GRAYSCALE};
use opencv::imgproc::{THRESH_OTSU, threshold};
use opencv::types::{VectorOfDMatch, VectorOfKeyPoint};
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

            let (k1, d1, k2, d2, matches) = matches(&files[0], &files[1]);

            let mat1 = dynamic_image_to_mat(&convert_to_dynamic_image(&files[0]), IMREAD_GRAYSCALE);
            let mat2 = dynamic_image_to_mat(&convert_to_dynamic_image(&files[1]), IMREAD_GRAYSCALE);

            let mut output =  Mat::default();
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
            ).unwrap();
            mat_to_dynamic_image(&output).save("output.tiff").unwrap();
        }
    };
}

/**
* 画像のサイズを変更する
 */
fn resize_image(img: DynamicImage, scale: f32) -> DynamicImage {
    let image1 = img.resize(img.width() * scale as u32, img.height() * scale as u32, image::imageops::FilterType::Nearest);
    return image1;
}

fn matches(f1: &str, f2: &str) -> (Vector<KeyPoint>, Mat, Vector<KeyPoint>, Mat, Vector<DMatch>) {
    let (k1, d1) = get_keypoints_and_descriptor(&convert_to_dynamic_image(f1));
    let (k2, d2) = get_keypoints_and_descriptor(&convert_to_dynamic_image(f2));

    let mut bf_matcher = BFMatcher::create(NORM_HAMMING, true).unwrap();

    let mut matches = VectorOfDMatch::new();
    bf_matcher
        .train_match(&d1, &d2, &mut matches, &no_array())
        .unwrap();

    let mut good_matches = VectorOfDMatch::new();
    for m in &matches {
        if m.distance < 20.0 {
            good_matches.push(m);
        }
    }

    println!("\n MATHES : {} --------------------", &matches.len());
    println!("GOOD MATHES : {} --------------------", &good_matches.len());
    return (k1, d1, k2, d2, good_matches);
}

fn get_keypoints_and_descriptor(image: &DynamicImage) -> (Vector<KeyPoint>, Mat) {
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
    return (key_points, descriptors);
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
