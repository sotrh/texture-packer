use std::{io::Read, ops::Deref, path::Path};

use image::EncodableLayout;
use wgpu::{util::DeviceExt, RenderPassColorAttachment};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RectVertex {
    position: glam::Vec2,
    uv: glam::Vec2,
}

pub struct RectRenderer {
    format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: CpuBuffer<glam::Mat4>,
    uniform_bg: wgpu::BindGroup,
    texture_layout: wgpu::BindGroupLayout,
    default_sampler: wgpu::Sampler,
}

impl RectRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let uniform_buffer = CpuBuffer::new(
            device,
            glam::Mat4::IDENTITY,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CameraBinder"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(
                        std::mem::size_of::<glam::Mat4>() as _
                    ),
                },
                count: None,
            }],
        });
        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.buffer.as_entire_binding(),
            }],
        });

        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let default_sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let module = device.create_shader_module(wgpu::include_wgsl!("shaders/rect.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&uniform_layout, &texture_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<RectVertex>() as _,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x2,
                        1 => Float32x2,
                    ],
                }],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::all(),
                })],
            }),
            multiview: None,
        });

        Self {
            format,
            pipeline,
            uniform_buffer,
            uniform_bg,
            texture_layout,
            default_sampler,
        }
    }

    pub fn bind_textures(&self, device: &wgpu::Device, textures: &[Texture]) -> TextureBindings {
        TextureBindings {
            bindings: textures
                .iter()
                .map(|t| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.texture_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&t.view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.default_sampler),
                            },
                        ],
                    })
                })
                .collect(),
        }
    }

    pub fn update_uniforms<C: Camera>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: &C,
    ) {
        self.uniform_buffer
            .update(device, queue, |data| *data = camera.view_matrix());
    }

    pub fn create_render_pass<'a: 'b, 'b>(
        encoder: &'a mut wgpu::CommandEncoder,
        view: &'a wgpu::TextureView,
    ) -> wgpu::RenderPass<'b> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        })
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        batch: &RectBatch,
        textures: &TextureBindings,
    ) {
        let mut pass = Self::create_render_pass(encoder, target);
        if batch.texture_indices.len() > 0 {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.uniform_bg, &[]);
            pass.set_vertex_buffer(0, batch.vertex_buffer.buffer.slice(..));
            pass.set_index_buffer(
                batch.index_buffer.buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            for (i, t) in batch.texture_indices.iter().enumerate() {
                // TODO: maybe check if the texture exists for this rect
                let texture = &textures.bindings[*t];
                pass.set_bind_group(1, texture, &[]);
                pass.draw_indexed(0..6, i as i32 * 4, 0..1);
            }
        }
    }
}

pub struct TextureBindings {
    bindings: Vec<wgpu::BindGroup>,
}

#[derive(Debug)]
pub struct Rect {
    pub position: glam::Vec2,
    pub size: glam::Vec2,
    pub texture: usize,
}

pub struct RectBatch {
    texture_indices: Vec<usize>,
    vertex_buffer: CpuBuffer<Vec<RectVertex>>,
    index_buffer: CpuBuffer<[u32; 6]>,
}

impl RectBatch {
    pub fn with_rects(
        device: &wgpu::Device,
        rects: &[Rect],
    ) -> Self {
        let vertices = rects
            .iter()
            .flat_map(|r| {
                let min = r.position;
                let max = r.position + r.size;
                [
                    RectVertex {
                        position: min,
                        uv: glam::vec2(0.0, 0.0),
                    },
                    RectVertex {
                        position: glam::vec2(min.x, max.y),
                        uv: glam::vec2(0.0, 1.0),
                    },
                    RectVertex {
                        position: max,
                        uv: glam::vec2(1.0, 1.0),
                    },
                    RectVertex {
                        position: glam::vec2(max.x, min.y),
                        uv: glam::vec2(1.0, 0.0),
                    },
                ]
            })
            .collect::<Vec<_>>();
        let vertex_buffer = CpuBuffer::new(
            device,
            vertices,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
        );

        let index_buffer = CpuBuffer::new(device, [0u32, 1, 2, 0, 2, 3], wgpu::BufferUsages::INDEX);

        let texture_indices = rects.iter().map(|r| r.texture).collect::<Vec<_>>();

        Self {
            texture_indices,
            vertex_buffer,
            index_buffer,
        }
    }
}

pub trait Camera {
    fn view_matrix(&self) -> glam::Mat4;
}

#[derive(Debug)]
pub struct Camera2d {
    position: glam::Vec2,
    size: glam::Vec2,
}

impl Camera2d {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            position: glam::Vec2::ZERO,
            size: glam::vec2(width, height),
        }
    }

    pub fn set_size(&mut self, width: f32, height: f32) {
        self.size.x = width;
        self.size.y = height;
    }
}

impl Camera for Camera2d {
    fn view_matrix(&self) -> glam::Mat4 {
        let min = self.position;
        let max = min + self.size;
        glam::Mat4::orthographic_rh(min.x, max.x, min.y, max.y, 0.0, 1.0)
    }
}

pub trait CanSlice {}

impl<T> CanSlice for Vec<T> {}

pub trait BufferData {
    fn as_bytes(&self) -> &[u8];
    fn from_bytes(&mut self, data: &[u8]);
    fn size_in_bytes(&self) -> u64;
    fn capacity_in_bytes(&self) -> u64;
}

impl<T: bytemuck::Pod + bytemuck::Zeroable> BufferData for Vec<T> {
    #[inline(always)]
    fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self)
    }

    #[inline(always)]
    fn from_bytes(&mut self, data: &[u8]) {
        self.clear();
        self.extend_from_slice(bytemuck::cast_slice(data));
    }

    #[inline(always)]
    fn size_in_bytes(&self) -> u64 {
        (self.len() * std::mem::size_of::<T>()) as _
    }

    #[inline(always)]
    fn capacity_in_bytes(&self) -> u64 {
        (self.capacity() * std::mem::size_of::<T>()) as _
    }
}

impl<T: bytemuck::Pod + bytemuck::Zeroable, const S: usize> BufferData for [T; S] {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self)
    }

    fn from_bytes(&mut self, data: &[u8]) {
        let data_t: &[T] = bytemuck::cast_slice(data);
        for (i, t) in data_t.iter().cloned().enumerate() {
            self[i] = t;
        }
    }

    fn size_in_bytes(&self) -> u64 {
        std::mem::size_of::<Self>() as _
    }

    fn capacity_in_bytes(&self) -> u64 {
        self.size_in_bytes()
    }
}

// TODO: Awaiting specialization to stablize
// impl<T: bytemuck::Pod + bytemuck::Zeroable> BufferData for T {
//     #[inline(always)]
//     default fn as_bytes(&self) -> &[u8] {
//         bytemuck::bytes_of(self)
//     }
// }

impl BufferData for glam::Mat4 {
    #[inline(always)]
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    #[inline(always)]
    fn from_bytes(&mut self, data: &[u8]) {
        *self = *bytemuck::from_bytes(data);
    }

    #[inline(always)]
    fn size_in_bytes(&self) -> u64 {
        std::mem::size_of::<Self>() as _
    }

    #[inline(always)]
    fn capacity_in_bytes(&self) -> u64 {
        self.size_in_bytes()
    }
}

pub struct CpuBuffer<T: BufferData> {
    data: T,
    buffer: wgpu::Buffer,
    buffer_size: u64,
    usage: wgpu::BufferUsages,
}

impl<T: BufferData> CpuBuffer<T> {
    pub fn new(device: &wgpu::Device, data: T, usage: wgpu::BufferUsages) -> Self {
        let contents = data.as_bytes();
        let (buffer, buffer_size) = if contents.len() == 0 {
            (
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: data.capacity_in_bytes(),
                    usage,
                    mapped_at_creation: false,
                }),
                data.capacity_in_bytes(),
            )
        } else {
            (
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents,
                    usage,
                }),
                contents.len() as _,
            )
        };
        Self {
            data,
            buffer,
            buffer_size,
            usage,
        }
    }

    #[inline(always)]
    fn flush(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        debug_assert!(self.data.capacity_in_bytes() >= self.data.size_in_bytes());
        if self.buffer_size < self.data.size_in_bytes() {
            self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: self.data.capacity_in_bytes(),
                usage: self.usage,
                // TODO: look into using mapped_at_creation instead of write_buffer when resizing
                mapped_at_creation: false,
            });
        }
        queue.write_buffer(&self.buffer, 0, self.data.as_bytes());
    }

    #[inline(always)]
    fn pull(&mut self, device: &wgpu::Device) {
        let slice = self.buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let view = slice.get_mapped_range();
        self.data.from_bytes(&view);
    }

    #[inline(always)]
    fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, mut f: impl FnMut(&mut T)) {
        f(&mut self.data);
        self.flush(device, queue);
    }

    #[inline(always)]
    fn multi_update<'a>(
        &'a mut self,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
    ) -> BufferHandle<'a, T> {
        BufferHandle::new(device, queue, self)
    }

    #[inline(always)]
    fn get(&self) -> &T {
        &self.data
    }

    fn size(&self) -> u64 {
        self.data.size_in_bytes()
    }
}

pub struct BufferHandle<'a, T: BufferData> {
    buffer: &'a mut CpuBuffer<T>,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    dirty: bool,
}

impl<'a, T: BufferData> BufferHandle<'a, T> {
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        buffer: &'a mut CpuBuffer<T>,
    ) -> Self {
        Self {
            buffer,
            dirty: false,
            device,
            queue,
        }
    }
}

impl<'a, T: BufferData> Drop for BufferHandle<'a, T> {
    fn drop(&mut self) {
        if self.dirty {
            self.buffer.flush(self.device, self.queue);
        }
    }
}

pub struct Texture {
    inner: wgpu::Texture,
    desc: wgpu::TextureDescriptor<'static>,
    view: wgpu::TextureView,
}

impl Texture {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Texture {
        let desc = wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
        };
        let inner = device.create_texture(&desc);
        let view = inner.create_view(&wgpu::TextureViewDescriptor::default());
        Self { inner, desc, view }
    }

    pub fn open<P: AsRef<Path>>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: P,
    ) -> Result<Self, crate::Error> {
        let img = image::open(&path)?.to_rgba8();
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let data = img.as_bytes();

        let desc = wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: img.width(),
                height: img.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        };
        let inner = device.create_texture_with_data(queue, &desc, data);
        let view = inner.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(Self { inner, desc, view })
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn size(&self) -> glam::Vec2 {
        glam::vec2(self.desc.size.width as _, self.desc.size.height as _)
    }

    pub fn area(&self) -> f32 {
        (self.desc.size.width * self.desc.size.height) as _
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let adapter = instance
            .enumerate_adapters(wgpu::Backends::all())
            .next()
            .unwrap();
        pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::default(),
                limits: wgpu::Limits::downlevel_webgl2_defaults(),
            },
            None,
        ))
        .unwrap()
    }

    fn encoder(device: &wgpu::Device) -> wgpu::CommandEncoder {
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None })
    }

    #[test]
    fn cpu_buffer_pull() {
        let (device, queue) = init();
        let src = CpuBuffer::new(
            &device,
            (0..32).collect::<Vec<u32>>(),
            wgpu::BufferUsages::COPY_SRC,
        );
        let mut dst = CpuBuffer::new(
            &device,
            (0..32).map(|_| 0).collect::<Vec<u32>>(),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );

        let mut encoder = encoder(&device);
        encoder.copy_buffer_to_buffer(&src.buffer, 0, &dst.buffer, 0, src.size());
        queue.submit(Some(encoder.finish()));

        dst.pull(&device);

        assert_eq!(src.data, dst.data);
    }

    #[test]
    fn cpu_buffer_flush() {
        let (device, queue) = init();
        let mut src = CpuBuffer::new(
            &device,
            (0..32).map(|_| 0).collect::<Vec<u32>>(),
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        let mut dst = CpuBuffer::new(
            &device,
            (0..32).map(|_| 0).collect::<Vec<u32>>(),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );

        for (i, x) in src.data.iter_mut().enumerate() {
            *x = i as _;
        }
        src.flush(&device, &queue);

        let mut encoder = encoder(&device);
        encoder.copy_buffer_to_buffer(&src.buffer, 0, &dst.buffer, 0, src.size());
        queue.submit(Some(encoder.finish()));

        dst.pull(&device);

        assert_eq!(src.data, dst.data);
    }
}
