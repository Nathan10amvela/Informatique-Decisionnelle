package com.example.ro.dto.requestDTO;

import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.Data;

import java.time.Instant;

@Data
public class FamilyTreeDTO {

    @NotNull(message = "Name cannot be null.")
    @Size(min = 1, max = 150, message = "Name must be between 1 and 150 characters.")
    private String name;

    @Size(max = 500, message = "Description cannot exceed 500 characters.")
    private String description;

    @NotNull(message = "Geographic origin cannot be null.")
    @Size(min = 1, max = 150, message = "Geographic origin must be between 1 and 150 characters.")
    private String geographicOrigin;

    @NotNull(message = "Creator cannot be null.")
    @Size(min = 1, max = 100, message = "Creator must be between 1 and 100 characters.")
    private String creator;


}