class ProjectError(Exception):
    """Base error for project."""
    pass

class InvalidInputError(ProjectError):
    """For invalid user inputs."""
    pass

class FileHandlingError(ProjectError):
    """For file-related errors."""
    pass 