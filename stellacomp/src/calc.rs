use crate::utils::{convert_to_dynamic_image, dynamic_image_to_mat};
use image::DynamicImage;
use opencv::core::{no_array, DMatch, KeyPoint, Mat, Vector, NORM_HAMMING};
use opencv::features2d::AKAZE_DescriptorType::DESCRIPTOR_MLDB;
use opencv::features2d::KAZE_DiffusivityType::DIFF_PM_G2;
use opencv::features2d::{BFMatcher, DescriptorMatcherTraitConst, Feature2DTrait, AKAZE};
use opencv::imgcodecs::IMREAD_GRAYSCALE;
use opencv::types::{VectorOfDMatch, VectorOfKeyPoint};

pub fn matches(
    f1: &str,
    f2: &str,
) -> (Vector<KeyPoint>, Mat, Vector<KeyPoint>, Mat, Vector<DMatch>) {
    let (k1, d1) = get_keypoints_and_descriptor(&convert_to_dynamic_image(f1));
    let (k2, d2) = get_keypoints_and_descriptor(&convert_to_dynamic_image(f2));

    let bf_matcher = BFMatcher::create(NORM_HAMMING, true).unwrap();

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
