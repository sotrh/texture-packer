use winit::{
    event::*,
    event_loop::{self, EventLoop},
    window::WindowBuilder,
};

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .enumerate_adapters(wgpu::Backends::all())
        .filter(|a| a.is_surface_supported(&surface))
        .next()
        .expect("There should be a valid adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            limits: wgpu::Limits::downlevel_webgl2_defaults(),
            ..Default::default()
        },
        None,
    ))
    .expect("There should be a valid device");

    let format = surface.get_supported_formats(&adapter)[0];
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: window.inner_size().width,
        height: window.inner_size().height,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
    };
    surface.configure(&device, &config);

    let mut renderer = texture_packer::render::RectRenderer::new(&device, format, 512);
    let mut camera = texture_packer::render::Camera2d::new(config.width as _, config.height as _);

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => control_flow.set_exit(),
        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                },
            ..
        } => control_flow.set_exit(),
        Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
            config.width = size.width;
            config.height = size.height;
            camera.set_size(config.width as _, config.height as _);
            renderer.update_uniforms(&device, &queue, &camera);
        }
        Event::RedrawRequested(_) => match surface.get_current_texture() {
            Ok(texture) => {
                let view = texture.texture.create_view(&Default::default());
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                renderer.render(&mut encoder, &view);
                queue.submit(Some(encoder.finish()));
                texture.present();
            }
            Err(wgpu::SurfaceError::Outdated) => {}
            Err(e) => {
                eprintln!("{}", e);
                control_flow.set_exit_with_code(1);
            }
        },
        _ => {}
    })
}
