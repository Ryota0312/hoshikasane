use clap::Parser;
use image::DynamicImage;
use opencv::calib3d::{estimate_affine_2d, RANSAC};
use opencv::core::{no_array, KeyPointTraitConst, Mat, MatTraitConst, Point2f, Scalar, Vector};
use opencv::imgcodecs::IMREAD_COLOR;
use opencv::imgproc::warp_affine;
use stellacomp::{
    average_composition, convert_to_dynamic_image, draw_match_points, dynamic_image_to_mat,
    lighten_composition, mat_to_dynamic_image, matches,
};

#[derive(clap::Subcommand, Clone, Debug)]
enum Mode {
    LightenComposite {
        #[arg(short, long, value_delimiter = ',')]
        file: Vec<String>,

        #[arg(short, long)]
        output: String,
    },
    AverageComposite {
        #[arg(short, long, value_delimiter = ',')]
        file: Vec<String>,

        #[arg(short, long)]
        output: String,
    },
    AffineConvert {
        #[arg(short, long)]
        base: String,

        #[arg(short, long)]
        target: String,
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
        Mode::LightenComposite { file, output } => {
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
                new_image = lighten_composition(&new_image, &image);
            }
            new_image.save(output).unwrap();
        }
        Mode::AverageComposite { file, output } => {
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
                new_image = average_composition(&new_image, &image);
            }
            new_image.save(output).unwrap();
        }
        Mode::AffineConvert { base, target } => {
            if base == "" {
                println!("Should specify base file.");
                return;
            }
            if target == "" {
                println!("Should specify target file.");
                return;
            }

            let (k1, d1, k2, d2, matches) = matches(&base, &target);

            let base_img = dynamic_image_to_mat(&convert_to_dynamic_image(&base), IMREAD_COLOR);
            let target_image =
                dynamic_image_to_mat(&convert_to_dynamic_image(&target), IMREAD_COLOR);

            draw_match_points(&k1, &k2, &matches, &base_img, &target_image);

            let mut apt1: Vector<Point2f> = Vector::new();
            let mut apt2: Vector<Point2f> = Vector::new();
            for m in matches {
                apt1.push(k1.get(m.query_idx as usize).unwrap().pt());
                apt2.push(k2.get(m.train_idx as usize).unwrap().pt());
            }

            let affine =
                estimate_affine_2d(&apt2, &apt1, &mut no_array(), RANSAC, 3.0, 2000, 0.99, 10)
                    .unwrap();
            println!("Affine : {:?}", affine);

            let mut converted = Mat::default();
            warp_affine(
                &target_image,
                &mut converted,
                &affine,
                target_image.size().unwrap(),
                1,
                0,
                Scalar::default(),
            )
            .unwrap();
            mat_to_dynamic_image(&converted)
                .save("converted.tiff")
                .unwrap();

            mat_to_dynamic_image(&base_img).save("base.tiff").unwrap();
            mat_to_dynamic_image(&target_image)
                .save("target.tiff")
                .unwrap();
        }
    };
}
