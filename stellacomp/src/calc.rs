use crate::utils::{convert_to_dynamic_image, dynamic_image_to_mat};
use image::{DynamicImage, GenericImageView, RgbImage};
use opencv::core::{no_array, DMatch, KeyPoint, Mat, Vector, NORM_HAMMING};
use opencv::features2d::AKAZE_DescriptorType::DESCRIPTOR_MLDB;
use opencv::features2d::KAZE_DiffusivityType::DIFF_PM_G2;
use opencv::features2d::{BFMatcher, DescriptorMatcherTraitConst, Feature2DTrait, AKAZE};
use opencv::imgcodecs::IMREAD_GRAYSCALE;
use opencv::imgproc::{threshold, THRESH_OTSU};
use opencv::types::{VectorOfDMatch, VectorOfKeyPoint};

pub fn matches(
    f1: &str,
    f2: &str,
) -> (Vector<KeyPoint>, Mat, Vector<KeyPoint>, Mat, Vector<DMatch>) {
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

pub fn get_keypoints_and_descriptor(image: &DynamicImage) -> (Vector<KeyPoint>, Mat) {
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
pub fn binarize(image: &DynamicImage) -> DynamicImage {
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
