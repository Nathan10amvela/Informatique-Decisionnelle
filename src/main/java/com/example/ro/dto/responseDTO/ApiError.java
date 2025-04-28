package com.example.ro.dto.responseDTO;

import lombok.Data;

@Data
public class ApiError {

    private String value;
    private Object data;
    private String text;
}
