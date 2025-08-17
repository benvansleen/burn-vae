{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    pre-commit-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-utils.follows = "flake-utils";
      };
    };

    crane.url = "github:ipetkov/crane";

    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      pre-commit-hooks,
      rust-overlay,
      crane,
      treefmt-nix,
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

        treefmtEval = pkgs: treefmt-nix.lib.evalModule pkgs ./treefmt.nix;

        rustToolchain = pkgs.rust-bin.selectLatestNightlyWith (
          toolchain:
          toolchain.default.override {
            extensions = [
              "rust-src"
              "rust-analyzer"
            ];
            targets = [ "wasm32-unknown-unknown" ];
          }
        );

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
          inherit (craneLib.crateNameFromCargoToml { inherit src; }) version;
        };

        cargo-src = [
          ./Cargo.toml
          ./Cargo.lock
          (craneLib.fileset.commonCargoSources ./.)
        ];
      in
      rec {
        packages = rec {
          default = train;
          train = craneLib.buildPackage (
            individualCrateArgs
            // {
              pname = "train";
              cargoExtraArgs = "-p train";
              src = lib.fileset.toSource {
                root = ./.;
                fileset = lib.fileset.unions cargo-src;
              };
            }
          );

          web = craneLib.buildPackage (
            individualCrateArgs
            // rec {
              pname = "web";
              src = lib.fileset.toSource {
                root = ./.;
                fileset = lib.fileset.unions (
                  cargo-src
                  ++ [
                    ./web/public
                    ./web/style
                    ./model_artifacts/model.bin
                  ]
                );
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
            }
          );
        };

        devShells = {
          default =
            with pkgs;
            mkShell {
              buildInputs = [
                self.checks.${system}.pre-commit-check.enabledPackages
              ]
              ++ (
                with lib;
                pipe self.packages.${system} [
                  (mapAttrsToList (
                    _name: value: [
                      value.buildInputs
                      value.nativeBuildInputs
                      value.propagatedBuildInputs
                    ]
                  ))
                  flatten
                ]
              );

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

        formatter = (treefmtEval pkgs).config.build.wrapper;
        checks = {
          pre-commit-check = pre-commit-hooks.lib.${pkgs.system}.run {
            src = ./.;
            hooks = {
              treefmt = {
                enable = true;
                packageOverrides.treefmt = self.outputs.formatter.${pkgs.system};
              };
            };
          };
        };
      }
    );
}
