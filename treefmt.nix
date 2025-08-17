{
  projectRootFile = "flake.nix";
  settings.global.excludes = [
    ".envrc"
    "model_artifacts/"
  ];

  programs = {
    nixfmt.enable = true;
    statix.enable = true;
    beautysh.enable = true;
    shellcheck.enable = true;
    rustfmt.enable = true;
  };

  # List of formatters available at https://github.com/numtide/treefmt-nix?tab=readme-ov-file#supported-programs
}
