import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        from pixel_world_model.cli import main
        sys.exit(main())
    from pixel_world_model.gui import main
    main()
