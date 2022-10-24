mod pack;
pub use pack::*;
pub mod render;

pub struct TexturePack {
    atlas: (),
    texture: render::Texture,
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
