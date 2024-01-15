use crate::generate::{generate_data, SpiralItem};
use flume::{bounded, Receiver, Sender};
use once_cell::sync::OnceCell;

static CHAN: OnceCell<(
    Sender<SpiralItem>,
    Receiver<SpiralItem>,
)> = OnceCell::new();
static T: OnceCell<[std::thread::JoinHandle<()>; 2]> =
    OnceCell::new();

pub fn init() -> Receiver<SpiralItem> {
    let (tx, rx) = CHAN.get_or_init(|| bounded(10_000_000));
    T.get_or_init(|| {
        std::array::from_fn(|_| {
            let tx = tx.clone();
            std::thread::spawn(move || {
                generate_data(1000, tx);
            })
        })
    });

    rx.clone()
}
