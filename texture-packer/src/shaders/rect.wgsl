struct RectVertex {
    @location(0)
    position: vec2<f32>,
    @location(1)
    uv: vec2<f32>,
}

struct FsData {
    @builtin(position)
    screen_pos: vec4<f32>,
    @location(0)
    uv: vec2<f32>,
}

struct Uniforms {
    view_matrix: mat4x4<f32>,
}

@group(0)
@binding(0)
var<uniform> uniforms: Uniforms;


@vertex
fn vs_main(in: RectVertex) -> FsData {
    let screen_pos = uniforms.view_matrix * vec4(in.position, 0.0, 1.0);
    return FsData(screen_pos, in.uv);
}

@fragment
fn fs_main(in: FsData) -> @location(0) vec4<f32> {
    return vec4(1.0);
}