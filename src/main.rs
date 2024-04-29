use clap::Parser;
use image::{DynamicImage, GenericImageView, RgbImage};
use rawler::imgop::develop::RawDevelop;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, value_delimiter = ',')]
    file: Vec<String>,

    #[arg(short, long)]
    output: String,
}


fn main() {
    let args = Args::parse();

    if args.file.len() > 2 {
        println!("Should specify more than 2 images.");
        return;
    }

    let first_image = convert_to_dynamic_image(&args.file[0]);
    let mut new_image: DynamicImage = first_image.clone();
    for f in &args.file[1..args.file.len()] {
        let image = convert_to_dynamic_image(f);
        new_image = lighten_composition_inner(&new_image, &image);
    }
    new_image.save(args.output).unwrap();
}


fn convert_to_dynamic_image(file_path: &str) -> DynamicImage {
    let raw_image = rawler::decode_file(file_path).unwrap();
    let dev = RawDevelop::default();
    let image = dev.develop_intermediate(&raw_image).unwrap().to_dynamic_image().unwrap();
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