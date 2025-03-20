use image::{codecs::hdr::HdrDecoder, DynamicImage};

fn main() {
    let dec = HdrDecoder::new(std::io::BufReader::new(
        std::fs::File::open("resources/background.hdr").unwrap(),
    ))
    .unwrap();
    dbg!(dec.metadata());

    let img = DynamicImage::from_decoder(dec).unwrap();
    let img = img.to_rgb32f();

    img.save("resources/background.exr").unwrap();
}
