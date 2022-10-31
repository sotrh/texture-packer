mod pack;
use std::path::Path;

pub use pack::*;
pub mod render;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Supplied path is not a directory")]
    NotADirectory,
    #[error("Unable to process file/folder")]
    IO(#[from] std::io::Error),
    #[error("Unable to process image")]
    Image(#[from] image::ImageError),
    #[error("Unsupported format")]
    UnsupportedFormat,
}

#[derive(Debug)]
pub enum PackingAlgorithm {
    Simple,
}

#[derive(Debug)]
pub struct PackOptions {
    pub size: u32,
    pub padding: u32,
    pub format: wgpu::TextureFormat,
    pub usages: wgpu::TextureUsages,
    pub algorithm: PackingAlgorithm,
}

impl Default for PackOptions {
    fn default() -> Self {
        Self {
            size: 1080,
            padding: 0,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usages: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            algorithm: PackingAlgorithm::Simple,
        }
    }
}

pub struct TexturePack {
    pub textures: Vec<render::Texture>,
}

impl TexturePack {
    pub fn pack_folder<P: AsRef<Path>>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: P,
        options: PackOptions,
    ) -> Result<Self, Error> {
        if !path.as_ref().is_dir() {
            return Err(Error::NotADirectory);
        }

        let mut in_textures = Vec::new();
        let mut rects = Vec::new();
        std::fs::read_dir(path)?
            .filter_map(|entry| match entry {
                Ok(entry) => {
                    let path = entry.path();
                    let ext = path.extension()?.to_str()?.to_lowercase();
                    match &ext[..] {
                        "png" | "jpg" | "jpeg" => {
                            match render::Texture::open(device, queue, &path) {
                                Ok(texture) => Some(texture),
                                Err(e) => {
                                    eprintln!("Unable to create texture for {:?}\n{}", path, e);
                                    None
                                }
                            }
                        }
                        _ => return None,
                    }
                }
                Err(e) => {
                    eprintln!("Unable to process file: {}", e);
                    None
                }
            })
            .enumerate()
            .for_each(|(i, t)| {
                rects.push(render::Rect {
                    position: glam::Vec2::ZERO,
                    size: t.size(),
                    texture: i,
                });
                in_textures.push(t);
            });

        let mut texture_change_indices = vec![0];
        match options.algorithm {
            PackingAlgorithm::Simple => {
                rects.sort_by(|a, b| a.size.x.partial_cmp(&b.size.x).unwrap());
                let mut position = glam::Vec2::ZERO;
                let mut max_height = 0.0;
                for (i, r) in rects.iter_mut().enumerate() {
                    if position.x + options.padding as f32 + r.size.x > options.size as f32 {
                        position.x = 0.0;
                        position.y += max_height + options.padding as f32;
                        max_height = 0.0;
                    }

                    if position.y + options.padding as f32 + r.size.y > options.size as f32 {
                        texture_change_indices.push(i);
                        position.x = 0.0;
                        position.y = 0.0;
                    }

                    r.position = position;
                    position.x += r.size.x + options.padding as f32;
                    max_height = max_height.max(r.size.y);
                }
            }
        }

        let camera = render::Camera2d::new(options.size as f32, options.size as f32);
        let mut renderer = render::RectRenderer::new(device, options.format);
        renderer.update_uniforms(device, queue, &camera);
        let bindings = renderer.bind_textures(device, &in_textures);

        println!("{:#?}", rects);
        let textures = (0..texture_change_indices.len())
            .map(|i| {
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                let i2 = i + 1;
                let range = if i2 < texture_change_indices.len() {
                    texture_change_indices[i]..texture_change_indices[i2]
                } else {
                    texture_change_indices[i]..rects.len()
                };
                let batch = render::RectBatch::with_rects(device, &rects[range]);
                let texture = render::Texture::new(
                    device,
                    options.size,
                    options.size,
                    options.format,
                    options.usages | wgpu::TextureUsages::RENDER_ATTACHMENT,
                );
                renderer.render(&mut encoder, &texture.view(), &batch, &bindings);
                queue.submit(Some(encoder.finish()));
                texture
            })
            .collect::<Vec<_>>();

        Ok(Self { textures })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
