use std::collections::HashMap;
use std::path::Path;

pub mod render;
pub use render::Rect;

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
    #[error("Serialization error")]
    Serialization(#[from] serde_json::Error),
}

#[derive(Debug)]
pub enum PackingAlgorithm {
    Simple,
}

#[derive(Debug)]
pub struct PackOptions {
    pub size: u32,
    pub padding: u32,
    pub name: String,
    pub format: wgpu::TextureFormat,
    pub usages: wgpu::TextureUsages,
    pub algorithm: PackingAlgorithm,
}

impl Default for PackOptions {
    fn default() -> Self {
        Self {
            size: 1080,
            padding: 0,
            name: "Atlas".to_owned(),
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usages: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            algorithm: PackingAlgorithm::Simple,
        }
    }
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct TextureRectHandle {
    index: usize,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct TextureRect {
    texture_index: usize,
    uv_min: glam::Vec2,
    uv_max: glam::Vec2,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct TextureAtlas {
    name: String,
    rect_map: HashMap<String, usize>,
    rects: Vec<Rect>,
    texture_names: Vec<String>,
}

pub struct TexturePack {
    pub atlas: TextureAtlas,
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
        let mut rect_names = Vec::new();
        std::fs::read_dir(path)?
            .filter_map(|entry| match entry {
                Ok(entry) => {
                    let path = entry.path();
                    let ext = path.extension()?.to_str()?.to_lowercase();
                    match &ext[..] {
                        "png" | "jpg" | "jpeg" => {
                            match render::Texture::open(device, queue, &path) {
                                Ok(texture) => Some((texture, path)),
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
            .for_each(|(i, (t, p))| {
                let name = p
                    .file_name()
                    .expect("No non-files should reach this point")
                    .to_str()
                    .expect("File name should contain valid UTF-8 characters")
                    .to_string();
                rect_names.push(name);
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
                let mut indices = (0..rects.len()).collect::<Vec<usize>>();
                indices.sort_by(|a, b| rects[*a].size.x.partial_cmp(&rects[*b].size.x).unwrap());
                rects = indices.iter().map(|i| rects[*i]).collect::<Vec<_>>();
                rect_names = indices.iter().map(|i| rect_names[*i].clone()).collect::<Vec<_>>();

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
        
        
        
        let texture_names = (0..textures.len()).map(|i| format!("{}-{}.png", options.name, i)).collect();
        let atlas = TextureAtlas {
            name: options.name,
            rect_map: rect_names.into_iter().enumerate().map(|(i, name)| (name, i)).collect(),
            rects,
            texture_names,
        };

        Ok(Self { textures, atlas })
    }

    // pub fn get_handle(&self, name: &str) -> Option<TextureRectHandle> {
    //     self.atlas
    //         .get(name)
    //         .map(|index| TextureRectHandle { index: *index })
    // }

    pub fn save<P: AsRef<Path>>(&self, device: &wgpu::Device, queue: &wgpu::Queue, path: P) -> Result<(), crate::Error> {
        if !path.as_ref().is_dir() {
            return Err(Error::NotADirectory)
        }
        
        let path = path.as_ref();
        let atlas_json = serde_json::to_string_pretty(&self.atlas)?;
        std::fs::write(path.join(format!("{}.json", self.atlas.name)), atlas_json)?;

        for (name, texture) in self.atlas.texture_names.iter().zip(&self.textures) {
            texture.save(device, queue, path.join(name))?;
        }

        Ok(())
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
