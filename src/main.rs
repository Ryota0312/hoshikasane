use image::{DynamicImage, GenericImageView, RgbImage};
use rawler::imgop::develop::RawDevelop;

fn main() {
    let raw_image_1 = rawler::decode_file("sample/01_m42_sample.CR3").unwrap();
    let raw_image_2 = rawler::decode_file("sample/02_m42_sample.CR3").unwrap();


    let dev = RawDevelop::default();
    let image1 = dev.develop_intermediate(&raw_image_1).unwrap().to_dynamic_image().unwrap();
    let image2 = dev.develop_intermediate(&raw_image_2).unwrap().to_dynamic_image().unwrap();

    let image_buffer = lighten_composition(image1, image2);
    image_buffer.save("sample/composite.tiff").unwrap();
}


/**
 * 画像を比較明合成する
 */
fn lighten_composition(image1: DynamicImage, image2: DynamicImage) -> DynamicImage {
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