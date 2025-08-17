{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    pre-commit-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    rust-overlay = {
      ## pin rust compiler to March 2024
      url = "github:oxalica/rust-overlay/a30facb";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
      };
    };

    crane.url = "github:ipetkov/crane";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      pre-commit-hooks,
      rust-overlay,
      crane,
    }:
    with flake-utils.lib;
    eachSystem allSystems (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };
        inherit (pkgs) lib;

        rustToolchain = pkgs.rust-bin.selectLatestNightlyWith(
          toolchain: toolchain.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
          targets = [ "wasm32-unknown-unknown" ];
        });

        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
        src = craneLib.cleanCargoSource ./.;

        commonArgs = {
          src = craneLib.cleanCargoSource ./.;
          strictDeps = false;
          propagatedBuildInputs = with pkgs; [
            (python311.withPackages (pypkg: [
              pypkg.scikit-learn
            ]))
          ];
        };
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;
        individualCrateArgs = commonArgs // {
          inherit cargoArtifacts;
          inherit (craneLib.crateNameFromCargoToml {inherit src;}) version;
        };
      in
        rec {
          packages = rec {
            train = craneLib.buildPackage (
              individualCrateArgs // {
                pname = "train";
                cargoExtraArgs = "-p train";
                src = lib.fileset.toSource {
                  root = ./.;
                  fileset = lib.fileset.unions [
                    ./Cargo.toml
                    ./Cargo.lock
                    (craneLib.fileset.commonCargoSources ./dataset)
                    (craneLib.fileset.commonCargoSources ./vae)
                    (craneLib.fileset.commonCargoSources ./train)
                  ];
                };
                }
          );

            web = craneLib.buildPackage (
              individualCrateArgs
              // rec {
                pname = "web";
                src = lib.fileset.toSource {
                  root = ./.;
                  fileset = lib.fileset.unions [
                    ./Cargo.toml
                    ./Cargo.lock
                    (craneLib.fileset.commonCargoSources ./web)
                    (craneLib.fileset.commonCargoSources ./dataset)
                    (craneLib.fileset.commonCargoSources ./vae)
                    (craneLib.fileset.commonCargoSources ./train)
                    (craneLib.fileset.commonCargoSources ./inference)
                    ./web/public
                    ./web/style
                    ./model_artifacts/model.bin
                  ];
                };

                buildPhaseCargoCommand = "cargo leptos build --release";
                doCheck = false;
                doNotPostBuildInstallCargoBinaries = true;
                installPhaseCommand = ''
                  mkdir -p $out/bin
                  cp target/release/${pname} $out/bin/
                  cp -r target/site $out/bin/
                  wrapProgram $out/bin/${pname} \
                    --set LEPTOS_SITE_ROOT $out/bin/site
                '';

                buildInputs = with pkgs; [
                  cargo-leptos
                  binaryen
                ];
                nativeBuildInputs = with pkgs; [
                  makeWrapper
                ];
            });
          };

          devShells = {
            default =
              with pkgs;
            mkShell {
              buildInputs = [
                self.checks.${system}.pre-commit-check.enabledPackages
                self.packages.${system}.train.buildInputs
                self.packages.${system}.train.nativeBuildInputs
                self.packages.${system}.train.propagatedBuildInputs

                self.packages.${system}.web.buildInputs
                self.packages.${system}.web.nativeBuildInputs
                self.packages.${system}.web.propagatedBuildInputs
              ];

              shellHook = lib.concatStringsSep "\n\n" [
                self.checks.${system}.pre-commit-check.shellHook
                ''
                  export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib"
                  export RUST_BACKTRACE=0
                  export CC="gcc"
                ''
              ];
            };
          };

          checks = {
            pre-commit-check = pre-commit-hooks.lib.${pkgs.system}.run {
              src = ./.;
              hooks = { };
            };
          };
        });
}
