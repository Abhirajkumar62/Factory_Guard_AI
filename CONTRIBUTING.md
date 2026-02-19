# Contributing to Factory Guard AI

Thank you for your interest in contributing to Factory Guard AI! This document provides guidelines and information for contributors.

## 🚀 How to Contribute

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/Factory_Guard_AI.git
   cd Factory_Guard_AI
   ```

3. **Set up the development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Code Standards

- **Python**: Follow PEP 8 style guidelines
- **Documentation**: Use Google-style docstrings
- **Testing**: Write unit tests for new features
- **Commits**: Use clear, descriptive commit messages

### Testing

Run the test suite before submitting changes:
```bash
pytest tests/ -v --cov=src
```

### Pull Request Process

1. **Update documentation** for any new features
2. **Add tests** for new functionality
3. **Ensure CI passes** (GitHub Actions)
4. **Create a Pull Request** with a clear description
5. **Wait for review** and address feedback

## 📋 Issue Guidelines

- **Bug reports**: Include steps to reproduce, expected vs actual behavior
- **Feature requests**: Describe the use case and benefits
- **Questions**: Check existing issues and documentation first

## 🏗️ Architecture Guidelines

- **Modular design**: Keep components loosely coupled
- **Error handling**: Implement proper exception handling
- **Logging**: Use Python's logging module consistently
- **Configuration**: Use YAML configs for environment-specific settings

## 📊 Performance Considerations

- **Model efficiency**: Optimize for real-time predictions
- **Memory usage**: Monitor memory consumption for large datasets
- **Scalability**: Design for horizontal scaling

## 🤝 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn and contribute

## 📞 Getting Help

- **Documentation**: Check the `docs/` folder and README
- **Issues**: Open a GitHub issue for questions
- **Discussions**: Use GitHub Discussions for general topics

Thank you for contributing to Factory Guard AI! 🎯