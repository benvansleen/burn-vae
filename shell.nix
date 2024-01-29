let
    rustOverlay = builtins.fetchTarball "https://github.com/oxalica/rust-overlay/archive/master.tar.gz";
    pkgs = import <nixpkgs> {
      overlays = [ (import rustOverlay) ];
    };
in
pkgs.mkShell rec {
    buildInputs = with pkgs; [
    # rust-bin.nightly.latest.default
    # rust-analyzer
    (rust-bin.selectLatestNightlyWith( toolchain: toolchain.default.override {
        extensions = [ "rust-src" "rust-analyzer" ];
        targets = [ "wasm32-unknown-unknown" ];
    }))
    pkg-config
    openssl
    zlib
    poetry
    python311
    python311Packages.scikit-learn
    vulkan-headers
    vulkan-loader
    vulkan-validation-layers
    ];
    
    shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"
    '';
    
    RUST_BACKTRACE = 0;
    CC = "gcc";
}