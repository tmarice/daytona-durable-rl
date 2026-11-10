{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  packages = with pkgs; [
    just
  ];

  languages = {
    python = {
      enable = true;
      package = pkgs.python314;
      uv.enable = true;
    };
  };
}
